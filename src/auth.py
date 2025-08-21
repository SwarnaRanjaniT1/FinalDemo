"""
Simple authentication module simulating Azure AD SSO.
Implements basic role-based access control as described in the security architecture.
"""

import streamlit as st
from typing import Dict, Any, Optional
import hashlib
import time


class SimpleAuth:
    """
    Simple authentication system simulating enterprise SSO.
    In production, this would integrate with Azure AD.
    """
    
    def __init__(self):
        """Initialize the authentication system."""
        # Demo user database (in production, this would be Azure AD)
        self.users = {
            "admin": {
                "password_hash": self._hash_password("admin123"),
                "role": "admin",
                "full_name": "System Administrator",
                "department": "IT",
                "permissions": ["read", "write", "admin", "metrics"]
            },
            "user": {
                "password_hash": self._hash_password("password"),
                "role": "employee",
                "full_name": "Demo User",
                "department": "General",
                "permissions": ["read"]
            },
            "manager": {
                "password_hash": self._hash_password("manager123"),
                "role": "manager",
                "full_name": "Department Manager",
                "department": "Operations",
                "permissions": ["read", "write", "metrics"]
            }
        }
        
        # Session timeout (in seconds)
        self.session_timeout = 3600  # 1 hour
    
    def _hash_password(self, password: str) -> str:
        """
        Hash a password using SHA-256.
        
        Args:
            password: Plain text password
            
        Returns:
            Hashed password
        """
        return hashlib.sha256(password.encode()).hexdigest()
    
    def authenticate(self, username: str, password: str) -> bool:
        """
        Authenticate a user with username and password.
        
        Args:
            username: User's username
            password: User's password
            
        Returns:
            True if authentication successful, False otherwise
        """
        if not username or not password:
            return False
        
        username = username.lower().strip()
        
        if username not in self.users:
            return False
        
        user_data = self.users[username]
        password_hash = self._hash_password(password)
        
        if password_hash == user_data["password_hash"]:
            # Store user session data
            st.session_state.user_data = user_data.copy()
            st.session_state.user_data["username"] = username
            st.session_state.login_time = time.time()
            return True
        
        return False
    
    def check_permission(self, permission: str) -> bool:
        """
        Check if current user has a specific permission.
        
        Args:
            permission: Permission to check
            
        Returns:
            True if user has permission, False otherwise
        """
        if not st.session_state.get('authenticated', False):
            return False
        
        user_data = st.session_state.get('user_data', {})
        user_permissions = user_data.get('permissions', [])
        
        return permission in user_permissions
    
    def get_user_role(self) -> Optional[str]:
        """
        Get the current user's role.
        
        Returns:
            User role or None if not authenticated
        """
        if not st.session_state.get('authenticated', False):
            return None
        
        user_data = st.session_state.get('user_data', {})
        return user_data.get('role')
    
    def get_user_info(self) -> Optional[Dict[str, Any]]:
        """
        Get current user information.
        
        Returns:
            User information dictionary or None if not authenticated
        """
        if not st.session_state.get('authenticated', False):
            return None
        
        user_data = st.session_state.get('user_data', {})
        
        # Remove sensitive information
        safe_user_data = user_data.copy()
        safe_user_data.pop('password_hash', None)
        
        return safe_user_data
    
    def check_session_timeout(self) -> bool:
        """
        Check if the current session has timed out.
        
        Returns:
            True if session is valid, False if timed out
        """
        if not st.session_state.get('authenticated', False):
            return False
        
        login_time = st.session_state.get('login_time', 0)
        current_time = time.time()
        
        if current_time - login_time > self.session_timeout:
            # Session timed out
            self.logout()
            return False
        
        return True
    
    def logout(self) -> None:
        """Log out the current user."""
        keys_to_remove = [
            'authenticated', 'user_data', 'username', 'login_time',
            'vector_store', 'rag_pipeline', 'chat_history'
        ]
        
        for key in keys_to_remove:
            if key in st.session_state:
                del st.session_state[key]
    
    def require_permission(self, permission: str) -> bool:
        """
        Decorator-like function to require a specific permission.
        Shows error message if user doesn't have permission.
        
        Args:
            permission: Required permission
            
        Returns:
            True if user has permission, False otherwise
        """
        if not self.check_permission(permission):
            user_role = self.get_user_role()
            st.error(f"Access denied. This feature requires '{permission}' permission. Current role: {user_role}")
            return False
        return True
    
    def get_rbac_info(self) -> Dict[str, Any]:
        """
        Get Role-Based Access Control information for display.
        
        Returns:
            RBAC information dictionary
        """
        user_info = self.get_user_info()
        
        if not user_info:
            return {"status": "Not authenticated"}
        
        return {
            "status": "Authenticated",
            "username": user_info.get('username', 'Unknown'),
            "role": user_info.get('role', 'Unknown'),
            "full_name": user_info.get('full_name', 'Unknown'),
            "department": user_info.get('department', 'Unknown'),
            "permissions": user_info.get('permissions', []),
            "session_valid": self.check_session_timeout()
        }
    
    @staticmethod
    def get_role_descriptions() -> Dict[str, str]:
        """
        Get descriptions of available roles.
        
        Returns:
            Dictionary mapping roles to descriptions
        """
        return {
            "admin": "Full system access including user management and system configuration",
            "manager": "Access to documents, metrics, and team management features",
            "employee": "Basic access to document search and chat functionality"
        }
    
    @staticmethod
    def get_permission_descriptions() -> Dict[str, str]:
        """
        Get descriptions of available permissions.
        
        Returns:
            Dictionary mapping permissions to descriptions
        """
        return {
            "read": "View documents and use chat functionality",
            "write": "Upload and manage documents",
            "admin": "System administration and user management",
            "metrics": "View performance metrics and analytics"
        }
