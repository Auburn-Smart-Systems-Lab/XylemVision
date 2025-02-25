## Prerequisites

- **Python 3.x**
- **Django** (tested with Django 3.2)
- **Pillow** (for image processing)
- (Optional) **djongo** or **mongoengine** if you plan to integrate MongoDB as your backend database.

## Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/mohtasimhadi/root-structural-analysis.git
   cd root-structural-analysis
   ```

2. **Create and Activate a Virtual Environment:**

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies:**

   ```bash
   pip install django pillow
   ```

   > If you decide to use MongoDB, install one of the following:
   >
   > For djongo (compatible with Django 3.x):
   >
   > ```bash
   > pip install djongo pymongo
   > ```
   >
   > Or, for MongoEngine:
   >
   > ```bash
   > pip install mongoengine
   > ```

4. **Configure the Database:**

   Update your project's `settings.py` with your desired database settings. For example, to use djongo:

   ```python
   DATABASES = {
       'default': {
           'ENGINE': 'djongo',
           'NAME': 'your_database_name',
       }
   }
   ```

   Or, if you’re using MongoEngine, set up your connection in `settings.py` or in a separate configuration file:

   ```python
   import mongoengine
   mongoengine.connect(db='your_database_name', host='localhost', port=27017)
   ```

5. **Run Migrations:**

   ```bash
   python manage.py makemigrations
   python manage.py migrate
   ```

## Usage

1. **Start the Development Server:**

   ```bash
   python manage.py runserver
   ```