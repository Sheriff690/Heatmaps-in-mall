from flask import Flask, request, jsonify

app = Flask(__name__)

# Sample data
users = [
    {"id": 1, "name": "Alice", "email": "alice@example.com"},
    {"id": 2, "name": "Bob", "email": "bob@example.com"}
]

# Home route
@app.route('/')
def index():
    return "Welcome to the API!"

# Get all users
@app.route('/users', methods=['GET'])
def get_users():
    return jsonify(users), 200

# Get user by ID
@app.route('/users/<int:user_id>', methods=['GET'])
def get_user(user_id):
    user = next((u for u in users if u['id'] == user_id), None)
    if user:
        return jsonify(user), 200
    else:
        return jsonify({"message": "User not found"}), 404

# Add a new user
@app.route('/users', methods=['POST'])
def add_user():
    new_user = request.json
    users.append(new_user)
    return jsonify(new_user), 201

# Update user by ID
@app.route('/users/<int:user_id>', methods=['PUT'])
def update_user(user_id):
    user = next((u for u in users if u['id'] == user_id), None)
    if user:
        user.update(request.json)
        return jsonify(user), 200
    else:
        return jsonify({"message": "User not found"}), 404

# Delete user by ID
@app.route('/users/<int:user_id>', methods=['DELETE'])
def delete_user(user_id):
    global users
    users = [u for u in users if u['id'] != user_id]
    return jsonify({"message": "User deleted"}), 204

if __name__ == '_main_':
    app.run(debug=True)