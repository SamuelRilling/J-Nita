#!/usr/bin/env python
"""
Simple script to run the J-Nita v5.0 web application
"""

from app import app

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_ENV') == 'development'
    app.run(debug=debug, host='0.0.0.0', port=port)
