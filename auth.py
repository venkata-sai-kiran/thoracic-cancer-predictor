from flask import Blueprint, request, jsonify, make_response, redirect, url_for, render_template, session, flash
from functools import wraps
import jwt
import datetime
import bcrypt
import os
from pymongo import MongoClient
from bson import ObjectId
import re

# Initialize MongoDB connection
mongo_uri = os.getenv('MONGO_URI', 'mongodb://localhost:27017/')
client = MongoClient(mongo_uri)
db = client.thoracic_surgery_db
users_collection = db.users

# Create Blueprint
auth = Blueprint('auth', __name__)

# JWT Secret Key
JWT_SECRET_KEY = os.getenv('JWT_SECRET_KEY', 'your_very_secure_secret_key_change_this_in_production')
JWT_EXPIRATION = datetime.timedelta(days=1)

# Password regex pattern for validation
PASSWORD_PATTERN = re.compile(r'^(?=.*[A-Za-z])(?=.*\d)[A-Za-z\d@$!%*#?&]{8,}$')

# Token required decorator
def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = None
        
        # Check if token exists in cookies
        if 'token' in request.cookies:
            token = request.cookies['token']
        
        if not token:
            return redirect(url_for('auth.login', next=request.url))
        
        try:
            # Decode the token
            data = jwt.decode(token, JWT_SECRET_KEY, algorithms=["HS256"])
            current_user = users_collection.find_one({'_id': ObjectId(data['user_id'])})
            
            if not current_user:
                return redirect(url_for('auth.login', next=request.url))
            
        except jwt.ExpiredSignatureError:
            return redirect(url_for('auth.login', next=request.url, msg="Token expired. Please log in again."))
        except jwt.InvalidTokenError:
            return redirect(url_for('auth.login', next=request.url, msg="Invalid token. Please log in again."))
        
        # Add user to the request context
        return f(current_user, *args, **kwargs)
    
    return decorated

@auth.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        # Get form data
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        
        # Validate inputs
        if not all([username, email, password, confirm_password]):
            flash('All fields are required', 'danger')
            return render_template('register.html')
        
        # Validate email domain
        if not email.endswith('@gmail.com'):
            flash('Email must be a valid Gmail address ending with @gmail.com', 'danger')
            return render_template('register.html')
        
        if password != confirm_password:
            flash('Passwords do not match', 'danger')
            return render_template('register.html')
        
        if not PASSWORD_PATTERN.match(password):
            flash('Password must be at least 8 characters and contain both letters and numbers', 'danger')
            return render_template('register.html')
        
        # Check if username or email already exists
        if users_collection.find_one({'username': username}):
            flash('Username already exists', 'danger')
            return render_template('register.html')
            
        if users_collection.find_one({'email': email}):
            flash('Email already exists', 'danger')
            return render_template('register.html')
        
        # Hash the password
        hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
        
        # Create new user
        new_user = {
            'username': username,
            'email': email,
            'password': hashed_password,
            'created_at': datetime.datetime.utcnow(),
            'role': 'user'  # Default role
        }
        
        # Insert user to database
        result = users_collection.insert_one(new_user)
        
        if result.inserted_id:
            flash('Registration successful! Please log in.', 'success')
            next_page = request.args.get('next')
            if next_page:
                return redirect(url_for('auth.login', next=next_page))
            return redirect(url_for('auth.login'))
        else:
            flash('Registration failed. Please try again.', 'danger')
    
    # GET request
    return render_template('register.html')

@auth.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        # Get form data
        username_or_email = request.form.get('username_or_email')
        password = request.form.get('password')
        
        # Find user by username or email
        user = users_collection.find_one({
            '$or': [
                {'username': username_or_email},
                {'email': username_or_email}
            ]
        })
        
        if not user:
            flash('Invalid username or password', 'danger')
            return render_template('login.html')
        
        # Check password
        if bcrypt.checkpw(password.encode('utf-8'), user['password']):
            # Generate token
            token = jwt.encode({
                'user_id': str(user['_id']),
                'username': user['username'],
                'exp': datetime.datetime.utcnow() + JWT_EXPIRATION
            }, JWT_SECRET_KEY, algorithm="HS256")
            
            next_page = request.args.get('next', url_for('home'))
            
            # Create response and set cookie
            response = make_response(redirect(next_page))
            response.set_cookie('token', token, httponly=True, max_age=JWT_EXPIRATION.total_seconds())
            
            # Store user info in session
            session['user_id'] = str(user['_id'])
            session['username'] = user['username']
            session['role'] = user.get('role', 'user')
            
            return response
        else:
            flash('Invalid username or password', 'danger')
    
    # GET request
    if 'msg' in request.args:
        flash(request.args.get('msg'), 'info')
    
    return render_template('login.html')

@auth.route('/logout')
def logout():
    # Clear session
    session.clear()
    
    # Create response and clear cookie
    response = make_response(redirect(url_for('home')))
    response.delete_cookie('token')
    
    flash('You have been logged out successfully', 'success')
    return response

@auth.route('/profile')
@token_required
def profile(current_user):
    return render_template('profile.html', user=current_user)

@auth.route('/change-password', methods=['POST'])
@token_required
def change_password(current_user):
    current_password = request.form.get('current_password')
    new_password = request.form.get('new_password')
    confirm_password = request.form.get('confirm_password')
    
    # Validate inputs
    if not all([current_password, new_password, confirm_password]):
        flash('All fields are required', 'danger')
        return redirect(url_for('auth.profile'))
    
    if new_password != confirm_password:
        flash('New passwords do not match', 'danger')
        return redirect(url_for('auth.profile'))
    
    if not PASSWORD_PATTERN.match(new_password):
        flash('Password must be at least 8 characters and contain both letters and numbers', 'danger')
        return redirect(url_for('auth.profile'))
    
    # Check current password
    if not bcrypt.checkpw(current_password.encode('utf-8'), current_user['password']):
        flash('Current password is incorrect', 'danger')
        return redirect(url_for('auth.profile'))
    
    # Hash the new password
    hashed_password = bcrypt.hashpw(new_password.encode('utf-8'), bcrypt.gensalt())
    
    # Update password in database
    users_collection.update_one(
        {'_id': current_user['_id']},
        {'$set': {'password': hashed_password}}
    )
    
    flash('Password updated successfully', 'success')
    return redirect(url_for('auth.profile'))

# Admin-only decorator
def admin_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'role' not in session or session['role'] != 'admin':
            flash('Access denied. Admin privileges required.', 'danger')
            return redirect(url_for('home'))
        return f(*args, **kwargs)
    return decorated

# Helper function to check if user is logged in
def is_logged_in():
    return 'user_id' in session

# Get current user
def get_current_user():
    if is_logged_in():
        user_id = session.get('user_id')
        user = users_collection.find_one({'_id': ObjectId(user_id)})
        return user
    return None 