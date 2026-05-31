import sys
import bcrypt
import sqlite3

db = sqlite3.connect("insurai_memory/insurai_sessions.db")

def list_users():
    rows = db.execute("SELECT username, created_at FROM users ORDER BY created_at").fetchall()
    if not rows:
        print("No users found.")
        return
    print(f"\n{'Username':<20} {'Created'}")
    print("-" * 50)
    for username, created_at in rows:
        print(f"{username:<20} {created_at[:19] if created_at else 'unknown'}")
    print()

def create_user(username, password):
    existing = db.execute("SELECT 1 FROM users WHERE username = ?", (username,)).fetchone()
    if existing:
        print(f"❌  User '{username}' already exists.")
        return
    hashed = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
    from datetime import datetime, timezone
    db.execute(
        "INSERT INTO users (username, password_hash, created_at) VALUES (?, ?, ?)",
        (username, hashed, datetime.now(timezone.utc).isoformat())
    )
    db.commit()
    print(f"✅  User '{username}' created.")

def delete_user(username):
    if username == "admin":
        print("❌  Cannot delete the admin account.")
        return
    db.execute("DELETE FROM users WHERE username = ?", (username,))
    db.commit()
    print(f"🗑️   User '{username}' deleted.")

def change_password(username, new_password):
    existing = db.execute("SELECT 1 FROM users WHERE username = ?", (username,)).fetchone()
    if not existing:
        print(f"❌  User '{username}' not found.")
        return
    hashed = bcrypt.hashpw(new_password.encode(), bcrypt.gensalt()).decode()
    db.execute("UPDATE users SET password_hash = ? WHERE username = ?", (hashed, username))
    db.commit()
    print(f"✅  Password updated for '{username}'.")

def prompt(label, secret=False):
    if secret:
        import getpass
        return getpass.getpass(label)
    return input(label).strip()

if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else None

    if cmd == "list":
        list_users()

    elif cmd == "create":
        username = sys.argv[2] if len(sys.argv) > 2 else prompt("Username: ")
        password = sys.argv[3] if len(sys.argv) > 3 else prompt("Password: ", secret=True)
        if not username or not password:
            print("Username and password are required.")
        else:
            create_user(username, password)

    elif cmd == "delete":
        username = sys.argv[2] if len(sys.argv) > 2 else prompt("Username to delete: ")
        create_user if False else delete_user(username)

    elif cmd == "password":
        username = sys.argv[2] if len(sys.argv) > 2 else prompt("Username: ")
        password = prompt("New password: ", secret=True)
        change_password(username, password)

    else:
        print("""
InsurAI User Management
-----------------------
python create_user.py list                        — list all users
python create_user.py create <username>           — create a new user (prompts for password)
python create_user.py delete <username>           — delete a user
python create_user.py password <username>         — change a user's password
""")
