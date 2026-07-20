"""
Authentication handlers for breeze buddy.

This module contains the business logic for authentication operations:
- User login (database users)
- S2S token generation (long-lived tokens for server-to-server auth)
- User info retrieval
- Logout handling
"""

from datetime import datetime, timedelta, timezone

import jwt as pyjwt
from fastapi import HTTPException, status

from app.api.security.breeze_buddy.rbac_token import rbac_token_manager
from app.core.config.static import (
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES,
)
from app.core.logger import logger
from app.core.security.password import DUMMY_PASSWORD_HASH, verify_password
from app.core.security.scope import resolve_merchant_ids, resolve_reseller_ids
from app.core.security.token_revocation import revoke_token
from app.database.accessor.breeze_buddy.users import get_user_by_username
from app.schemas import (
    LoginRequest,
    S2STokenRequest,
    S2STokenResponse,
    TokenResponse,
    UserInfo,
    UserRole,
)


async def login_handler(
    login_request: LoginRequest,
) -> TokenResponse:
    """
    Handle user login with JWT token-based authentication.

    Supports database users with RBAC.

    Args:
        login_request: Login credentials (username, password)

    Returns:
        TokenResponse with access_token, token_type, expires_in

    Raises:
        HTTPException: 401 if credentials are invalid or account is inactive
    """
    user = await get_user_by_username(login_request.username)

    if user:
        # Verify password first to prevent username enumeration
        # Both is_active and password checks return identical error messages
        if not verify_password(login_request.password, user.password_hash):
            logger.warning(f"Failed login attempt for user: {login_request.username}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid username or password",
            )

        # Check account status after password verification
        if not user.is_active:
            logger.warning(f"Inactive user attempted login: {login_request.username}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid username or password",
            )

        # Resolve wildcard scopes to actual values before creating token.
        # This ensures the frontend receives concrete reseller_ids and
        # merchant_ids instead of ["*"] which it can't interpret.
        resolved_reseller_ids = user.reseller_ids
        resolved_merchant_ids = user.merchant_ids

        if user.role != UserRole.ADMIN:
            # Build a temporary UserInfo to use resolve functions
            temp_user_info = UserInfo(
                id=user.id,
                username=user.username,
                role=user.role,
                email=user.email,
                reseller_ids=user.reseller_ids,
                merchant_ids=user.merchant_ids,
                permissions=[],
                owner_id=user.owner_id,
            )

            # Resolve reseller_ids wildcards
            resolved_r = await resolve_reseller_ids(temp_user_info)
            if resolved_r is not None:
                resolved_reseller_ids = resolved_r
            # else: None means admin-created with truly unrestricted access,
            # keep original ["*"] — only admin-created accounts can reach here

            # Resolve merchant_identifiers wildcards
            resolved_m = await resolve_merchant_ids(temp_user_info)
            if resolved_m is not None:
                resolved_merchant_ids = resolved_m
            # else: None means admin-created with truly unrestricted access,
            # keep original ["*"] — only admin-created accounts can reach here

        # Generate JWT token with resolved RBAC data
        access_token = rbac_token_manager.create_access_token_with_rbac(
            user_id=user.id,
            username=user.username,
            role=user.role,
            reseller_ids=resolved_reseller_ids,
            merchant_ids=resolved_merchant_ids,
            email=user.email,
            owner_id=user.owner_id,
        )

        # Calculate expires_in based on JWT_ACCESS_TOKEN_EXPIRE_MINUTES
        expires_in = JWT_ACCESS_TOKEN_EXPIRE_MINUTES * 60  # Convert to seconds

        logger.info(
            f"Successful login for database user: {user.username} "
            f"(role: {user.role}, resellers: {user.reseller_ids}, shops: {user.merchant_ids})"
        )

        return TokenResponse(
            access_token=access_token,
            token_type="Bearer",
            expires_in=expires_in,
        )

    # Authentication failed: no such user. Still spend one bcrypt against a
    # dummy hash so an unknown username can't be distinguished from a wrong
    # password by response timing (username-enumeration oracle, PT-16).
    verify_password(login_request.password, DUMMY_PASSWORD_HASH)
    logger.warning(f"Failed login attempt for unknown user: {login_request.username}")
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid username or password",
    )


async def generate_s2s_token_handler(request: S2STokenRequest) -> S2STokenResponse:
    """
    Generate long-lived token for Server-to-Server (S2S) authentication.

    Allows admin users to generate long-lived JWT tokens (up to 365 days)
    for automated integrations and S2S communication.

    Security restrictions:
    - Only admin users can generate S2S tokens
    - Maximum token lifetime: 365 days

    Args:
        request: S2STokenRequest containing username, password, and token_lifetime_days

    Returns:
        S2STokenResponse with access_token, expires_in, expires_at

    Raises:
        HTTPException: 401 if credentials are invalid or account is inactive
        HTTPException: 403 if user is not an admin
    """
    # Authenticate user
    user = await get_user_by_username(request.username)
    if not user:
        # Spend one bcrypt against a dummy hash before failing so an unknown
        # username is timing-indistinguishable from a wrong password — this path
        # would otherwise reveal whether an admin account exists (PT-16).
        verify_password(request.password, DUMMY_PASSWORD_HASH)
        logger.warning(f"S2S token request failed: user not found - {request.username}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials"
        )

    # Verify password
    if not verify_password(request.password, user.password_hash):
        logger.warning(
            f"S2S token request failed: invalid password - {request.username}"
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials"
        )

    # Check if user is active
    if not user.is_active:
        logger.warning(f"S2S token request failed: inactive user - {request.username}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Account is inactive. Please contact administrator.",
        )

    # Security restriction: Only admins can generate S2S tokens
    if user.role != UserRole.ADMIN:
        logger.warning(
            f"S2S token request denied: non-admin user - {request.username} (role: {user.role})"
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only admin users can generate S2S tokens. Please contact your administrator.",
        )

    # Generate long-lived token
    expires_delta = timedelta(days=request.token_lifetime_days)

    # Start with the admin's own scopes
    resolved_reseller_ids = user.reseller_ids
    resolved_merchant_ids = user.merchant_ids

    # Apply caller-requested scope restrictions.
    # Prevent escalation: requested scopes must be a subset of the admin's own scopes.
    if request.reseller_ids is not None:
        if "*" not in resolved_reseller_ids:
            invalid = set(request.reseller_ids) - set(resolved_reseller_ids)
            if invalid:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Cannot grant reseller access beyond your own scope: {sorted(invalid)}",
                )
        resolved_reseller_ids = request.reseller_ids

    if request.merchant_ids is not None:
        if "*" not in resolved_merchant_ids:
            invalid = set(request.merchant_ids) - set(resolved_merchant_ids)
            if invalid:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Cannot grant merchant access beyond your own scope: {sorted(invalid)}",
                )
        resolved_merchant_ids = request.merchant_ids

    # If scopes are restricted (not wildcard), downgrade the token role
    # from ADMIN to RESELLER. RBAC checks short-circuit on role == "admin"
    # before inspecting reseller_ids/merchant_ids, so an ADMIN-role token
    # would bypass those restrictions entirely.
    scopes_are_restricted = (
        "*" not in resolved_reseller_ids or "*" not in resolved_merchant_ids
    )
    token_role = UserRole.RESELLER if scopes_are_restricted else user.role

    access_token = rbac_token_manager.create_access_token_with_rbac(
        user_id=user.id,
        username=user.username,
        role=token_role,
        reseller_ids=resolved_reseller_ids,
        merchant_ids=resolved_merchant_ids,
        email=user.email,
        owner_id=user.owner_id,
        expires_delta=expires_delta,
    )

    # Calculate response metadata
    expires_in_seconds = int(expires_delta.total_seconds())
    expires_at = (datetime.now(timezone.utc) + expires_delta).isoformat()

    logger.info(
        f"S2S token generated for user: {user.username} "
        f"(auth_role: {user.role}, token_role: {token_role}, "
        f"lifetime: {request.token_lifetime_days} days)"
    )

    return S2STokenResponse(
        success=True,
        access_token=access_token,
        token_type="Bearer",
        expires_in=expires_in_seconds,
        expires_at=expires_at,
        note=(
            f"Long-lived S2S token valid for {request.token_lifetime_days} days "
            f"(effective role: {token_role.value}). "
            f"Store securely and rotate before expiration!"
        ),
    )


async def get_user_info_handler(current_user: UserInfo) -> UserInfo:
    """
    Get current authenticated user information from JWT token.

    Args:
        current_user: UserInfo extracted from JWT token

    Returns:
        UserInfo object with user details
    """
    logger.info(
        f"User info request from {current_user.username} (role: {current_user.role})"
    )
    return current_user


async def logout_handler(token: str) -> dict:
    """
    Handle logout for JWT token-based authentication.

    The token is added to the server-side revocation denylist (keyed by a hash
    of the token, with a TTL equal to its remaining lifetime), so it can no
    longer authenticate even though its signature is still valid (PT-22).

    Returns:
        Success message.
    """
    try:
        payload = pyjwt.decode(
            token,
            rbac_token_manager.jwt_manager.secret_key,
            algorithms=[rbac_token_manager.jwt_manager.algorithm],
            options={"verify_exp": False},
        )
        exp = int(payload.get("exp", 0))
        revoked = await revoke_token(token, exp)
    except Exception as e:
        logger.error(f"Logout: failed to revoke token: {e}")
        revoked = False

    logger.info(f"Logout endpoint called (token revoked={revoked})")

    return {
        "success": True,
        "message": "Logout successful. Token has been revoked server-side.",
        "revoked": revoked,
    }
