## Prerequisites

- **Python 3.x**

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
   pip install -r requirements.txt
   ```

4. **Run Migrations:**

   ```bash
   python manage.py makemigrations
   python manage.py migrate
   ```

## Usage

1. **Start the Development Server:**

   ```bash
   python manage.py runserver
   ```

   ```bash
   docker build -t root-structural-analysis .
   docker run -p 8000:8000 root-structural-analysis
   ```