import pandas as pd
import os
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class AuthHandler:
    """Handle user authentication against customer.csv"""
    
    def __init__(self, base_path):
        self.base_path = base_path
        self.customer_file = os.path.join(base_path, "Database", "customer.csv")
        self._load_customer_data()
    
    def _load_customer_data(self):
        """Load customer data from CSV file"""
        try:
            if not os.path.exists(self.customer_file):
                raise FileNotFoundError(f"Customer file not found: {self.customer_file}")
            
            self.customers_df = pd.read_csv(self.customer_file)
            
            # Clean column names (remove extra spaces)
            self.customers_df.columns = self.customers_df.columns.str.strip()
            
            logger.info(f"Loaded {len(self.customers_df)} customer records")
            logger.info(f"Customer columns: {list(self.customers_df.columns)}")
            
        except Exception as e:
            logger.error(f"Error loading customer data: {str(e)}")
            raise
    
    def authenticate(self, nik, password):
        """
        Authenticate user with NIK and password
        
        Args:
            nik (str): User's NIK
            password (str): User's password
            
        Returns:
            dict: User data if authentication successful, None otherwise
        """
        try:
            # Clean inputs
            nik = str(nik).strip()
            password = str(password).strip()
            
            # Find user by NIK and password
            user_row = self.customers_df[
                (self.customers_df['NIK'].astype(str) == nik) & 
                (self.customers_df['password'].astype(str) == password)
            ]
            
            if len(user_row) == 0:
                logger.warning(f"Authentication failed for NIK: {nik}")
                return None
            
            # Convert to dictionary
            user_data = user_row.iloc[0].to_dict()
            
            # Update last login (in a real system, you'd update the CSV/database)
            user_data['last_login'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            logger.info(f"User authenticated successfully: {user_data['name']} (NIK: {nik})")
            
            return user_data
            
        except Exception as e:
            logger.error(f"Error during authentication: {str(e)}")
            return None
    
    def get_user_by_id(self, customer_id):
        """
        Get user data by customer ID
        
        Args:
            customer_id (str): Customer ID
            
        Returns:
            dict: User data if found, None otherwise
        """
        try:
            user_row = self.customers_df[self.customers_df['customer_id'] == customer_id]
            
            if len(user_row) == 0:
                return None
            
            return user_row.iloc[0].to_dict()
            
        except Exception as e:
            logger.error(f"Error getting user by ID: {str(e)}")
            return None
    
    def is_valid_user(self, customer_id):
        """
        Check if customer ID is valid
        
        Args:
            customer_id (str): Customer ID to validate
            
        Returns:
            bool: True if valid, False otherwise
        """
        try:
            return customer_id in self.customers_df['customer_id'].values
        except Exception as e:
            logger.error(f"Error validating user: {str(e)}")
            return False
