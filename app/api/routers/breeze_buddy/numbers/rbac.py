"""
RBAC (Role-Based Access Control) utilities for outbound numbers.
Only admin users can manage outbound numbers.
"""

from typing import List, Optional

from fastapi import HTTPException, status

from app.core.logger import logger
from app.schemas import UserInfo


def validate_number_access(
    current_user: UserInfo,
    reseller_id: Optional[str],
    merchant_id: Optional[str],
    operation: str = "access",
) -> None:
    """
    Validate the caller may access an outbound number given its ownership.

    Fail-closed: a number with a null reseller_id is reachable only by admin,
    and a null merchant_id (reseller pool number) only by callers scoped to the
    reseller. This mirrors templates/rbac.py and does NOT repeat the
    null-merchant-skip pattern flagged by PT-15.
    """
    if current_user.role == "admin":
        return

    if not reseller_id or (
        reseller_id not in current_user.reseller_ids
        and "*" not in current_user.reseller_ids
    ):
        logger.warning(
            f"User {current_user.username} attempted to {operation} outbound number "
            f"for unauthorized reseller: {reseller_id}"
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Access denied to reseller {reseller_id}",
        )

    if (
        merchant_id
        and merchant_id not in current_user.merchant_ids
        and "*" not in current_user.merchant_ids
    ):
        logger.warning(
            f"User {current_user.username} attempted to {operation} outbound number "
            f"for unauthorized merchant: {merchant_id}"
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Access denied to merchant {merchant_id}",
        )


def require_admin_access(
    current_user: UserInfo, operation: str = "perform this operation"
) -> None:
    """
    Validate user is an admin.

    Outbound numbers are system-wide resources that require admin access.

    Args:
        current_user: Current authenticated user
        operation: Operation being performed (for error message)

    Raises:
        HTTPException: 403 if user is not admin
    """
    if current_user.role != "admin":
        logger.warning(
            f"Non-admin user {current_user.username} (role: {current_user.role}) "
            f"attempted to {operation}"
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Admin access required to {operation}",
        )


def filter_numbers_by_rbac(numbers: List, current_user: UserInfo) -> List:
    """
    Filter outbound numbers based on user's RBAC permissions.

    Currently all authenticated users can view outbound numbers,
    but only admins can create/update/delete.

    Args:
        numbers: List of outbound number objects
        current_user: Current authenticated user

    Returns:
        List of outbound numbers the caller is scoped to (PT-13). Previously a
        no-op that returned every tenant's numbers to any authenticated user.
    """
    if current_user.role == "admin":
        return numbers

    if "*" in current_user.reseller_ids and "*" in current_user.merchant_ids:
        return numbers

    filtered = []
    for number in numbers:
        has_reseller_access = bool(number.reseller_id) and (
            "*" in current_user.reseller_ids
            or number.reseller_id in current_user.reseller_ids
        )
        # Reseller-pool numbers (null merchant_id) stay visible to anyone scoped
        # to the reseller — they are needed to understand call routing — but
        # numbers with a null reseller_id are dropped for non-admins (fail-closed).
        has_merchant_access = (not number.merchant_id) or (
            "*" in current_user.merchant_ids
            or number.merchant_id in current_user.merchant_ids
        )
        if has_reseller_access and has_merchant_access:
            filtered.append(number)

    return filtered
