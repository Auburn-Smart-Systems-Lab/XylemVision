#!/usr/bin/env python
import os
import sys
import socket

def get_local_ip():
    """Get the local IP address of the current machine."""
    try:
        # This connects to a dummy IP to get the interface IP without sending data
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"

def main():
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'root.settings')
    
    # Print local IP address for easy access
    print(f"Your local IP address is: {get_local_ip()}")
    print("Run the server with: python manage.py runserver 0.0.0.0:8000 to listen on all interfaces")
    
    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Couldn't import Django. Are you sure it's installed and "
            "available on your PYTHONPATH environment variable? Did you "
            "forget to activate a virtual environment?"
        ) from exc
    execute_from_command_line(sys.argv)


if __name__ == '__main__':
    main()
