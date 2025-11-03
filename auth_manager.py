import streamlit as st
import os
import supabase as sb
from dotenv import load_dotenv
from typing import Optional, Dict, Any
import time


# Load environment variables
load_dotenv()

# Supabase configuration
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY")

if not SUPABASE_URL or not SUPABASE_ANON_KEY:
    st.error("Supabase credentials not found in environment variables")
    st.stop()

# Initialize Supabase client
try:
    supabase: sb.Client = sb.create_client(SUPABASE_URL, SUPABASE_ANON_KEY)
except Exception as e:
    st.error(f"Failed to initialize Supabase client: {e}")
    st.stop()

class AuthManager:
    """Production-ready Supabase Authentication Manager"""
    
    def __init__(self):
        self.supabase = supabase
    
    def sign_up(self, email: str, password: str) -> Dict[str, Any]:
        """Sign up a new user with email confirmation disabled for development"""
        try:
            response = self.supabase.auth.sign_up({
                "email": email,
                "password": password,
                "options": {
                    "email_redirect_to": None  # Disable email confirmation for now
                }
            })
            
            if response.user:
                return {
                    "success": True, 
                    "data": response,
                    "message": "Account created successfully! You can now login."
                }
            else:
                return {
                    "success": False, 
                    "error": "Failed to create account. Email might already exist."
                }
                
        except Exception as e:
            error_msg = str(e)
            if "already registered" in error_msg.lower():
                return {"success": False, "error": "Email already registered. Please login instead."}
            elif "invalid email" in error_msg.lower():
                return {"success": False, "error": "Please enter a valid email address."}
            elif "password" in error_msg.lower():
                return {"success": False, "error": "Password must be at least 6 characters long."}
            else:
                return {"success": False, "error": f"Registration failed: {error_msg}"}
    
    def sign_in(self, email: str, password: str) -> Dict[str, Any]:
        """Sign in existing user"""
        try:
            response = self.supabase.auth.sign_in_with_password({
                "email": email,
                "password": password
            })
            
            if response.user:
                return {"success": True, "data": response}
            else:
                return {"success": False, "error": "Invalid email or password"}
                
        except Exception as e:
            error_msg = str(e).lower()
            if "invalid login credentials" in error_msg:
                return {"success": False, "error": "Invalid email or password"}
            elif "too many requests" in error_msg:
                return {"success": False, "error": "Too many login attempts. Please wait a few minutes."}
            elif "email not confirmed" in error_msg:
                return {"success": False, "error": "Please confirm your email address before logging in"}
            else:
                return {"success": False, "error": f"Login failed: {str(e)}"}
    
    def sign_out(self) -> Dict[str, Any]:
        """Sign out current user"""
        try:
            response = self.supabase.auth.sign_out()
            return {"success": True, "data": response}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def get_current_user(self) -> Optional[Dict[str, Any]]:
        """Get current authenticated user"""
        try:
            response = self.supabase.auth.get_user()
            if response and response.user:
                return {
                    "id": response.user.id,
                    "email": response.user.email,
                    "created_at": str(response.user.created_at),
                    "email_confirmed": response.user.email_confirmed_at is not None
                }
            return None
        except Exception:
            return None
    
    def get_session(self) -> Optional[Dict[str, Any]]:
        """Get current session"""
        try:
            response = self.supabase.auth.get_session()
            if response and response.access_token:
                return {
                    "access_token": response.access_token,
                    "refresh_token": response.refresh_token,
                    "expires_at": response.expires_at
                }
            return None
        except Exception:
            return None

# Initialize auth manager
auth_manager = AuthManager()

def init_session_state():
    """Initialize session state variables"""
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    if 'user_data' not in st.session_state:
        st.session_state.user_data = None
    if 'auth_checked' not in st.session_state:
        st.session_state.auth_checked = False

def check_authentication() -> tuple[bool, Optional[Dict[str, Any]]]:
    """Check if user is authenticated"""
    init_session_state()
    
    # If we haven't checked auth yet or user is not authenticated, check Supabase
    if not st.session_state.auth_checked or not st.session_state.authenticated:
        user_data = auth_manager.get_current_user()
        session_data = auth_manager.get_session()
        
        if user_data and session_data:
            st.session_state.authenticated = True
            st.session_state.user_data = user_data
            st.session_state.auth_checked = True
            return True, user_data
        else:
            st.session_state.authenticated = False
            st.session_state.user_data = None
            st.session_state.auth_checked = True
            return False, None
    
    # Return cached result
    return st.session_state.authenticated, st.session_state.user_data

def handle_login(email: str, password: str) -> bool:
    """Handle login process"""
    if not email or not password:
        st.error("Please enter both email and password")
        return False
    
    with st.spinner("Signing in..."):
        result = auth_manager.sign_in(email, password)
    
    if result["success"]:
        # Update session state
        user_data = {
            "id": result["data"].user.id,
            "email": result["data"].user.email,
            "created_at": str(result["data"].user.created_at)
        }
        
        st.session_state.authenticated = True
        st.session_state.user_data = user_data
        st.session_state.auth_checked = True
        
        st.success("Login successful!")
        time.sleep(1)  # Brief pause for user feedback
        st.rerun()
        return True
    else:
        st.error(result['error'])
        return False

def handle_signup(email: str, password: str, confirm_password: str) -> bool:
    """Handle signup process"""
    if not email or not password or not confirm_password:
        st.error("Please fill in all fields")
        return False
    
    if password != confirm_password:
        st.error("Passwords do not match")
        return False
    
    if len(password) < 6:
        st.error("Password must be at least 6 characters long")
        return False
    
    with st.spinner("Creating account..."):
        result = auth_manager.sign_up(email, password)
    
    if result["success"]:
        st.success("Account created successfully! You can now login.")
        return True
    else:
        st.error(result['error'])
        return False

def handle_logout():
    """Handle logout process"""
    result = auth_manager.sign_out()
    
    # Clear session state regardless of API result
    st.session_state.authenticated = False
    st.session_state.user_data = None
    st.session_state.auth_checked = False
    
    if result["success"]:
        st.success("Logged out successfully!")
    else:
        st.info("Logged out locally")
    
    time.sleep(1)
    st.rerun()

def display_auth_ui():
    """Display authentication UI in sidebar"""
    init_session_state()
    is_authenticated, user_data = check_authentication()
    
    st.sidebar.markdown("---")
    
    if is_authenticated and user_data:
        # User is logged in - show profile
        st.sidebar.markdown("### 👤 User Profile")
        st.sidebar.markdown(f"**Email:** {user_data['email']}")
        st.sidebar.markdown(f"**Status:** Logged in")
        
        if st.sidebar.button("Logout", use_container_width=True, key="logout_btn"):
            handle_logout()
    
    else:
        # User not logged in - show auth forms
        st.sidebar.markdown("### 🔐 Authentication")
        
        # Toggle between login and signup
        if 'show_signup' not in st.session_state:
            st.session_state.show_signup = False
        
        if st.session_state.show_signup:
            # Signup form
            st.sidebar.markdown("#### Sign Up")
            
            with st.sidebar.form("signup_form"):
                email = st.text_input("Email", placeholder="Enter your email")
                password = st.text_input("Password", type="password", placeholder="Create a password")
                confirm_password = st.text_input("Confirm Password", type="password")
                
                col1, col2 = st.columns(2)
                with col1:
                    signup_btn = st.form_submit_button("Sign Up", use_container_width=True)
                with col2:
                    if st.form_submit_button("Login Instead", use_container_width=True):
                        st.session_state.show_signup = False
                        st.rerun()
                
                if signup_btn:
                    if handle_signup(email, password, confirm_password):
                        st.session_state.show_signup = False
                        st.rerun()
        
        else:
            # Login form
            st.sidebar.markdown("#### Login")
            
            with st.sidebar.form("login_form"):
                email = st.text_input("Email", placeholder="Enter your email")
                password = st.text_input("Password", type="password", placeholder="Enter your password")
                
                col1, col2 = st.columns(2)
                with col1:
                    login_btn = st.form_submit_button("Login", use_container_width=True)
                with col2:
                    if st.form_submit_button("Sign Up", use_container_width=True):
                        st.session_state.show_signup = True
                        st.rerun()
                
                if login_btn:
                    handle_login(email, password)
    
    return is_authenticated, user_data

def require_auth(show_message: bool = True):
    """Require authentication to proceed"""
    is_authenticated, user_data = check_authentication()
    
    if not is_authenticated and show_message:
        st.warning("Please login to access this feature")
        st.stop()
    
    return is_authenticated, user_data

def get_current_user_id() -> Optional[str]:
    """Get current user ID"""
    is_authenticated, user_data = check_authentication()
    if is_authenticated and user_data:
        return user_data.get('id')
    return None

def get_current_user_email() -> Optional[str]:
    """Get current user email"""
    is_authenticated, user_data = check_authentication()
    if is_authenticated and user_data:
        return user_data.get('email')
    return None

# Main function to initialize auth system
def init_auth() -> tuple[bool, Optional[Dict[str, Any]]]:
    """Initialize authentication system - call this in your main app"""
    return display_auth_ui()