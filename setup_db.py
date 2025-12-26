import mysql.connector
from db_config import DB_CONFIG

def setup_database():
    try:
        # Connect to MySQL Server (excluding database to create it if needed)
        conn = mysql.connector.connect(
            host=DB_CONFIG['host'],
            user=DB_CONFIG['user'],
            password=DB_CONFIG['password']
        )
        cursor = conn.cursor()

        # Create Database
        cursor.execute(f"CREATE DATABASE IF NOT EXISTS {DB_CONFIG['database']}")
        print(f"Database '{DB_CONFIG['database']}' check/creation successful.")

        # Connect to the specific database
        conn.database = DB_CONFIG['database']

        # Create Table
        create_table_query = """
        CREATE TABLE IF NOT EXISTS attendance (
            id INT AUTO_INCREMENT PRIMARY KEY,
            name VARCHAR(255) NOT NULL,
            time TIME NOT NULL,
            date DATE NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        """
        cursor.execute(create_table_query)
        print("Table 'attendance' check/creation successful.")

        conn.commit()
        cursor.close()
        conn.close()
        print("Database setup completed successfully.")

    except mysql.connector.Error as err:
        print(f"Error: {err}")
        print("Please ensure MySQL is running and credentials in db_config.py are correct.")

if __name__ == "__main__":
    setup_database()
