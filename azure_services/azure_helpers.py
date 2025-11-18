# azure_helpers.py
import os
import streamlit as st
from urllib.parse import quote_plus
from azure.identity import ClientSecretCredential
from azure.storage.blob import BlobServiceClient
from azure.mgmt.datafactory import DataFactoryManagementClient
from sqlalchemy import create_engine, inspect
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# ==================== CACHED HELPER FUNCTIONS ====================
# These are standalone functions that can be cached by Streamlit
# Instance methods will call these functions

@st.cache_resource(show_spinner=False, ttl=3600)
def _get_sql_engine_cached():
    """Cached function to create SQLAlchemy engine"""
    try:
        # Try to get from Streamlit secrets first, fallback to environment variables
        try:
            sql_user = st.secrets.get('AZURE_SQL_USER')
            sql_password = st.secrets.get('AZURE_SQL_PASSWORD')
            sql_server = st.secrets.get('AZURE_SQL_SERVER')
            sql_database = st.secrets.get('AZURE_SQL_DATABASE')
        except:
            pass
        
        if not sql_user:
            sql_user = os.getenv('AZURE_SQL_USER')
        if not sql_password:
            sql_password = os.getenv('AZURE_SQL_PASSWORD')
        if not sql_server:
            sql_server = os.getenv('AZURE_SQL_SERVER')
        if not sql_database:
            sql_database = os.getenv('AZURE_SQL_DATABASE')
        
        # Validate required variables
        required_vars = {
            'AZURE_SQL_USER': sql_user,
            'AZURE_SQL_PASSWORD': sql_password,
            'AZURE_SQL_SERVER': sql_server,
            'AZURE_SQL_DATABASE': sql_database
        }
        missing_vars = [var for var, value in required_vars.items() if not value]
        
        if missing_vars:
            raise ValueError(f"Missing required configuration: {', '.join(missing_vars)}")
        
        # URL encode the password to handle special characters
        encoded_password = quote_plus(sql_password)
        
        # Build connection string with Azure SQL Database specific parameters
        connection_string = (
            f"mssql+pyodbc://{sql_user}:"
            f"{encoded_password}@"
            f"{sql_server}/"
            f"{sql_database}"
            f"?driver=ODBC+Driver+17+for+SQL+Server"
            f"&Encrypt=yes"
            f"&TrustServerCertificate=no"
            f"&Connection+Timeout=30"
        )
        
        # Create engine with connection timeout
        engine = create_engine(
            connection_string,
            connect_args={
                "timeout": 30,
                "autocommit": False
            },
            pool_pre_ping=True  # Verify connections before using
        )
        return engine
    except Exception as e:
        print(f"Error creating SQL engine: {e}")
        return None

@st.cache_data(show_spinner=False, ttl=300)
def _get_all_schemas_cached():
    """Cached function to get all schemas"""
    try:
        engine = _get_sql_engine_cached()
        if engine is None:
            print("SQL engine is None, cannot get schemas")
            return ['dbo']  # Return default schema as fallback
        
        inspector = inspect(engine)
        schemas = inspector.get_schema_names()
        return [s for s in schemas if s not in ['information_schema', 'sys']]
    except Exception as e:
        print(f"Error getting schemas: {e}")
        return ['dbo']  # Return default schema as fallback

@st.cache_data(show_spinner=False, ttl=300)
def _get_tables_by_schema_cached(schema_name):
    """Cached function to get tables by schema"""
    try:
        engine = _get_sql_engine_cached()
        if engine is None:
            print("SQL engine is None, cannot get tables")
            return ['FactVisit', 'DimPatient', 'DimDoctor', 'DimHospital', 'DimDate', 'DimMedication']
        
        inspector = inspect(engine)
        tables = inspector.get_table_names(schema=schema_name)
        return tables
    except Exception as e:
        print(f"Error getting tables: {e}")
        return ['FactVisit', 'DimPatient', 'DimDoctor', 'DimHospital', 'DimDate', 'DimMedication']

@st.cache_data(show_spinner=False, ttl=300)
def _get_table_schema_cached(schema_name, table_name):
    """Cached function to get table schema"""
    try:
        engine = _get_sql_engine_cached()
        if engine is None:
            print("SQL engine is None, cannot get table schema")
            return {}
        
        inspector = inspect(engine)
        columns = inspector.get_columns(table_name, schema=schema_name)
        
        schema_info = {}
        for col in columns:
            schema_info[col['name']] = {
                'type': str(col['type']),
                'nullable': col['nullable']
            }
        return schema_info
    except Exception as e:
        print(f"Error getting table schema: {e}")
        return {}

@st.cache_data(show_spinner=False, ttl=600)
def _read_csv_from_blob_cached(container_name, blob_path):
    """Cached function to read CSV from blob"""
    try:
        # Get storage credentials
        try:
            storage_account = st.secrets.get('AZURE_STORAGE_ACCOUNT')
            storage_key = st.secrets.get('AZURE_STORAGE_KEY')
        except:
            pass
        
        if not storage_account:
            storage_account = os.getenv('AZURE_STORAGE_ACCOUNT')
        if not storage_key:
            storage_key = os.getenv('AZURE_STORAGE_KEY')
        
        connection_string = (
            f"DefaultEndpointsProtocol=https;"
            f"AccountName={storage_account};"
            f"AccountKey={storage_key};"
            f"EndpointSuffix=core.windows.net"
        )
        
        blob_client = BlobServiceClient.from_connection_string(connection_string)
        container_client = blob_client.get_container_client(container_name)
        blob_client_file = container_client.get_blob_client(blob_path)
        
        download_stream = blob_client_file.download_blob()
        csv_data = download_stream.readall()
        
        if not csv_data or len(csv_data) == 0:
            print("Error reading CSV from blob: empty file")
            return None

        # Try multiple decoding and parsing strategies for robustness
        from io import StringIO
        decode_attempts = ['utf-8', 'utf-8-sig', 'latin-1']
        sep_attempts = [None, ',', ';', '\t']  # None enables auto-detect with python engine

        last_error = None
        for enc in decode_attempts:
            try:
                text = csv_data.decode(enc, errors='replace')
            except Exception as e:
                last_error = e
                continue

            for sep in sep_attempts:
                try:
                    df = pd.read_csv(
                        StringIO(text),
                        sep=sep,
                        engine='python' if sep is None else 'python',
                        on_bad_lines='skip'
                    )
                    # Basic sanity: must have at least 1 column
                    if df is not None and df.shape[1] > 0:
                        return df
                except Exception as e:
                    last_error = e
                    continue

        print(f"Error reading CSV from blob: {last_error}")
        return None
    except Exception as e:
        print(f"Error reading CSV from blob: {e}")
        return None

@st.cache_data(show_spinner=False, ttl=300)
def _list_csv_files_in_blob_cached(container_name, folder_path):
    """Cached function to list CSV files in blob"""
    try:
        # Get storage credentials
        try:
            storage_account = st.secrets.get('AZURE_STORAGE_ACCOUNT')
            storage_key = st.secrets.get('AZURE_STORAGE_KEY')
        except:
            pass
        
        if not storage_account:
            storage_account = os.getenv('AZURE_STORAGE_ACCOUNT')
        if not storage_key:
            storage_key = os.getenv('AZURE_STORAGE_KEY')
        
        connection_string = (
            f"DefaultEndpointsProtocol=https;"
            f"AccountName={storage_account};"
            f"AccountKey={storage_key};"
            f"EndpointSuffix=core.windows.net"
        )
        
        blob_client = BlobServiceClient.from_connection_string(connection_string)
        container_client = blob_client.get_container_client(container_name)
        
        csv_files = []
        blobs = container_client.list_blobs(name_starts_with=folder_path)
        
        for blob in blobs:
            if blob.name.endswith('.csv'):
                csv_files.append(blob.name)
        
        return csv_files
    except Exception as e:
        print(f"Error listing CSV files: {e}")
        return []

class AzureServices:
    def __init__(self):
        # Try to get from Streamlit secrets first, fallback to environment variables
        try:
            # Check if Streamlit secrets are available
            if hasattr(st, 'secrets') and st.secrets is not None:
                self.tenant_id = st.secrets.get('AZURE_TENANT_ID') or os.getenv('AZURE_TENANT_ID')
                self.client_id = st.secrets.get('AZURE_CLIENT_ID') or os.getenv('AZURE_CLIENT_ID')
                self.client_secret = st.secrets.get('AZURE_CLIENT_SECRET') or os.getenv('AZURE_CLIENT_SECRET')
                self.subscription_id = st.secrets.get('AZURE_SUBSCRIPTION_ID') or os.getenv('AZURE_SUBSCRIPTION_ID')
            else:
                # Streamlit secrets not available, use environment variables
                self.tenant_id = os.getenv('AZURE_TENANT_ID')
                self.client_id = os.getenv('AZURE_CLIENT_ID')
                self.client_secret = os.getenv('AZURE_CLIENT_SECRET')
                self.subscription_id = os.getenv('AZURE_SUBSCRIPTION_ID')
        except (AttributeError, KeyError, Exception):
            # Fallback to environment variables if Streamlit secrets not available
            self.tenant_id = os.getenv('AZURE_TENANT_ID')
            self.client_id = os.getenv('AZURE_CLIENT_ID')
            self.client_secret = os.getenv('AZURE_CLIENT_SECRET')
            self.subscription_id = os.getenv('AZURE_SUBSCRIPTION_ID')
        
        # Validate credentials before creating ClientSecretCredential
        if not self.tenant_id or not self.client_id or not self.client_secret:
            missing = []
            if not self.tenant_id:
                missing.append('AZURE_TENANT_ID')
            if not self.client_id:
                missing.append('AZURE_CLIENT_ID')
            if not self.client_secret:
                missing.append('AZURE_CLIENT_SECRET')
            raise ValueError(
                f"Missing required Azure credentials: {', '.join(missing)}. "
                f"Please set these as environment variables or in Streamlit secrets."
            )
        
        self.credential = ClientSecretCredential(
            tenant_id=self.tenant_id,
            client_id=self.client_id,
            client_secret=self.client_secret
        )
    
    # ==================== BLOB STORAGE OPERATIONS ====================
    
    def get_blob_service_client(self):
        """Get Blob Service Client"""
        try:
            storage_account = st.secrets.get('AZURE_STORAGE_ACCOUNT')
            storage_key = st.secrets.get('AZURE_STORAGE_KEY')
        except:
            pass
        
        if not storage_account:
            storage_account = os.getenv('AZURE_STORAGE_ACCOUNT')
        if not storage_key:
            storage_key = os.getenv('AZURE_STORAGE_KEY')
            
        connection_string = (
            f"DefaultEndpointsProtocol=https;"
            f"AccountName={storage_account};"
            f"AccountKey={storage_key};"
            f"EndpointSuffix=core.windows.net"
        )
        return BlobServiceClient.from_connection_string(connection_string)
    
    def list_csv_files_in_blob(self, container_name, folder_path):
        """List all CSV files in blob storage folder"""
        # Use cached function
        return _list_csv_files_in_blob_cached(container_name, folder_path)
    
    def upload_csv_to_blob(self, container_name, folder_path, file_data, file_name):
        """Upload CSV file to blob storage"""
        try:
            blob_client = self.get_blob_service_client()
            container_client = blob_client.get_container_client(container_name)
            
            # Ensure container exists
            try:
                container_client.get_container_properties()
            except Exception:
                # Container doesn't exist, create it
                blob_client.create_container(container_name)
            
            # Construct blob path with folder
            blob_path = f"{folder_path}/{file_name}" if folder_path else file_name
            
            # Upload file
            blob_client_file = container_client.get_blob_client(blob_path)
            blob_client_file.upload_blob(file_data, overwrite=True)
            
            return True, f"File uploaded successfully to {blob_path}"
        except Exception as e:
            print(f"Error uploading CSV to blob: {e}")
            return False, f"Error uploading file: {str(e)}"
    
    def read_csv_from_blob(self, container_name, blob_path):
        """Read CSV file from blob storage"""
        # Use cached function
        return _read_csv_from_blob_cached(container_name, blob_path)
    
    # ==================== SQL DATABASE OPERATIONS ====================
    
    def get_sql_engine(self):
        """Create SQLAlchemy engine for SQL Database"""
        # Use cached function
        return _get_sql_engine_cached()
    
    def test_sql_connection(self):
        """Test SQL Database connection"""
        try:
            engine = self.get_sql_engine()
            if engine is None:
                return False, "Failed to create SQL engine - Check configuration"
            
            # Try to connect with detailed error handling
            with engine.connect() as conn:
                result = conn.execute("SELECT 1 as test")
                return True, "✅ Connection successful! Database is accessible."
        except Exception as e:
            error_msg = str(e)
            error_code = None
            
            # Extract error code if present
            if "53" in error_msg:
                error_code = "53"
            elif "40615" in error_msg:
                error_code = "40615"
            elif "18456" in error_msg:
                error_code = "18456"
            
            # Provide detailed error messages
            if error_code == "53":
                return False, (
                    "❌ Network Error (53): Server not found or not accessible.\n\n"
                    "🔧 Solutions:\n"
                    "1. Check Azure SQL Database firewall rules\n"
                    "2. Add your IP address to firewall exceptions\n"
                    "3. Verify server name: dataiq-server.database.windows.net\n"
                    "4. Check if corporate firewall blocks port 1433"
                )
            elif error_code == "40615":
                return False, (
                    "❌ Firewall Error (40615): Your IP is not allowed.\n\n"
                    "🔧 Solutions:\n"
                    "1. Go to Azure Portal → SQL Server → Firewall\n"
                    "2. Add your current IP address\n"
                    "3. Or enable 'Allow Azure services' temporarily"
                )
            elif error_code == "18456":
                return False, (
                    "❌ Authentication Error (18456): Login failed.\n\n"
                    "🔧 Solutions:\n"
                    "1. Verify username: dataiq-serveradmin\n"
                    "2. Check password is correct\n"
                    "3. Ensure user account exists and is active"
                )
            elif "Login timeout" in error_msg or "timeout" in error_msg.lower():
                return False, (
                    "❌ Connection Timeout: Could not reach the server.\n\n"
                    "🔧 Solutions:\n"
                    "1. Check network connectivity\n"
                    "2. Verify firewall rules\n"
                    "3. Check if server is running"
                )
            else:
                return False, f"❌ Connection failed: {error_msg}\n\nTip: Check Azure Portal firewall settings and verify credentials."
    
    def debug_connection_info(self):
        """Debug connection information"""
        try:
            sql_user = st.secrets.get('AZURE_SQL_USER')
            sql_password = st.secrets.get('AZURE_SQL_PASSWORD')
            sql_server = st.secrets.get('AZURE_SQL_SERVER')
            sql_database = st.secrets.get('AZURE_SQL_DATABASE')
        except:
            pass
        
        if not sql_user:
            sql_user = os.getenv('AZURE_SQL_USER')
        if not sql_password:
            sql_password = os.getenv('AZURE_SQL_PASSWORD')
        if not sql_server:
            sql_server = os.getenv('AZURE_SQL_SERVER')
        if not sql_database:
            sql_database = os.getenv('AZURE_SQL_DATABASE')
        
        try:
            return {
                'sql_user': sql_user,
                'sql_password': '***' if sql_password else None,
                'sql_server': sql_server,
                'sql_database': sql_database,
                'connection_string': f"mssql+pyodbc://{sql_user}:***@{sql_server}/{sql_database}?driver=ODBC+Driver+17+for+SQL+Server"
            }
        except Exception as e:
            return {'error': str(e)}
    
    def test_network_connectivity(self):
        """Test basic network connectivity to Azure SQL Server"""
        import socket
        try:
            sql_server = st.secrets.get('AZURE_SQL_SERVER') or os.getenv('AZURE_SQL_SERVER')
            if not sql_server:
                return False, "No server configured"
            
            # Test DNS resolution
            try:
                ip = socket.gethostbyname(sql_server)
            except socket.gaierror:
                return False, f"DNS resolution failed for {sql_server}"
            
            # Test port connectivity (1433 for SQL Server)
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(10)
            result = sock.connect_ex((ip, 1433))
            sock.close()
            
            if result == 0:
                return True, f"Network connectivity OK: {sql_server} ({ip})"
            else:
                return False, f"Port 1433 not accessible on {sql_server} ({ip}) - Check firewall settings"
        except Exception as e:
            return False, f"Network test failed: {str(e)}"
    
    def get_current_ip(self):
        """Get current public IP address for firewall configuration"""
        try:
            import urllib.request
            import json
            
            # Try multiple services to get IP
            services = [
                'https://api.ipify.org?format=json',
                'https://ifconfig.me/ip',
                'https://icanhazip.com'
            ]
            
            for service in services:
                try:
                    with urllib.request.urlopen(service, timeout=5) as response:
                        if 'json' in service:
                            data = json.loads(response.read().decode())
                            return True, data.get('ip', 'Unknown')
                        else:
                            ip = response.read().decode().strip()
                            return True, ip
                except:
                    continue
            
            return False, "Could not determine IP address"
        except Exception as e:
            return False, f"Error getting IP: {str(e)}"
    
    def get_all_schemas(self):
        """Get all schemas from SQL Database"""
        # Use cached function
        return _get_all_schemas_cached()
    
    def get_tables_by_schema(self, schema_name):
        """Get all tables in a specific schema"""
        # Use cached function
        return _get_tables_by_schema_cached(schema_name)
    
    def get_table_schema(self, schema_name, table_name):
        """Get column information for a specific table"""
        # Use cached function
        return _get_table_schema_cached(schema_name, table_name)
    
    # ==================== ADF DEPLOYMENT AND PIPELINE OPERATIONS ====================
    
    def get_adf_client(self):
        """Get Data Factory Management Client"""
        try:
            return DataFactoryManagementClient(self.credential, self.subscription_id)
        except Exception as e:
            print(f"Error creating ADF client: {e}")
            return None
    
    def deploy_generated_code(self, generated_code):
        """Execute and deploy the generated code to ADF"""
        try:
            import re
            
            # Import necessary Azure SDK modules that the generated code might need
            try:
                from azure.identity import ClientSecretCredential
                from azure.mgmt.datafactory import DataFactoryManagementClient
                from azure.mgmt.datafactory.models import (
                    PipelineResource, ExecuteDataFlowActivity, ActivityPolicy,
                    ActivityDependency, DataFlowReference,
                    ExecuteDataFlowActivityTypePropertiesCompute,
                    LinkedServiceResource, DatasetResource, DataFlowResource,
                    MappingDataFlow
                )
                # Try to import AzureMLExecutePipelineActivity if it exists
                try:
                    from azure.mgmt.datafactory.models import AzureMLExecutePipelineActivity
                except ImportError:
                    AzureMLExecutePipelineActivity = None
            except ImportError as e:
                return False, f"Missing required Azure SDK imports: {str(e)}"
            
            # Fix common code generation issues before execution
            # Replace AzureMLExecutePipelineActivity with ExecuteDataFlowActivity if it's incorrectly used
            fixed_code = generated_code
            if "AzureMLExecutePipelineActivity" in generated_code:
                # This is a code generation error - AzureMLExecutePipelineActivity should be ExecuteDataFlowActivity
                # Auto-fix by replacing all instances
                import re
                # Replace AzureMLExecutePipelineActivity with ExecuteDataFlowActivity
                fixed_code = re.sub(r'AzureMLExecutePipelineActivity', 'ExecuteDataFlowActivity', fixed_code)
                print("Auto-fixed: Replaced 'AzureMLExecutePipelineActivity' with 'ExecuteDataFlowActivity'")
            
            # Create a local namespace for executing the code with necessary imports
            local_namespace = {
                '__builtins__': __builtins__,
                'ClientSecretCredential': ClientSecretCredential,
                'DataFactoryManagementClient': DataFactoryManagementClient,
                'PipelineResource': PipelineResource,
                'ExecuteDataFlowActivity': ExecuteDataFlowActivity,
                'ActivityPolicy': ActivityPolicy,
                'ActivityDependency': ActivityDependency,
                'DataFlowReference': DataFlowReference,
                'ExecuteDataFlowActivityTypePropertiesCompute': ExecuteDataFlowActivityTypePropertiesCompute,
                'LinkedServiceResource': LinkedServiceResource,
                'DatasetResource': DatasetResource,
                'DataFlowResource': DataFlowResource,
                'MappingDataFlow': MappingDataFlow,
            }
            
            # Add AzureMLExecutePipelineActivity to namespace if available
            if AzureMLExecutePipelineActivity:
                local_namespace['AzureMLExecutePipelineActivity'] = AzureMLExecutePipelineActivity
            
            # Execute the generated code to define the class
            exec(fixed_code, local_namespace, local_namespace)
            
            # Find the pipeline class (usually named *Pipeline or *CSVToSQLPipeline)
            # Exclude Azure SDK model classes like PipelineResource, DatasetResource, etc.
            excluded_classes = {'PipelineResource', 'DatasetResource', 'DataFlowResource', 
                              'LinkedServiceResource', 'ExecuteDataFlowActivity', 'ActivityPolicy',
                              'ActivityDependency', 'DataFlowReference', 'MappingDataFlow',
                              'ExecuteDataFlowActivityTypePropertiesCompute', 'ClientSecretCredential',
                              'DataFactoryManagementClient', 'AzureMLExecutePipelineActivity'}
            
            pipeline_class = None
            # First, look for classes with deploy_complete_solution method (highest priority)
            for name, obj in local_namespace.items():
                if isinstance(obj, type) and name not in excluded_classes:
                    if hasattr(obj, 'deploy_complete_solution'):
                        pipeline_class = obj
                        break
            
            # If not found, look for classes with 'Pipeline' or 'CSV' in name
            if pipeline_class is None:
                for name, obj in local_namespace.items():
                    if isinstance(obj, type) and name not in excluded_classes:
                        if ('Pipeline' in name or 'CSV' in name) and not name.endswith('Resource'):
                            # Also check if it has methods that suggest it's a management class
                            if hasattr(obj, 'create_pipeline') or hasattr(obj, 'deploy') or hasattr(obj, '__init__'):
                                pipeline_class = obj
                                break
            
            # Final fallback: look for any class that looks like a pipeline management class
            if pipeline_class is None:
                for name, obj in local_namespace.items():
                    if isinstance(obj, type) and name not in excluded_classes:
                        # Check if it has init with typical pipeline parameters
                        import inspect
                        try:
                            sig = inspect.signature(obj.__init__)
                            params = list(sig.parameters.keys())
                            # Look for typical pipeline init parameters
                            if 'subscription_id' in params or 'factory_name' in params or 'resource_group' in params:
                                pipeline_class = obj
                                break
                        except:
                            pass
            
            if pipeline_class is None:
                return False, "Could not find pipeline class in generated code. Please ensure the code contains a class with 'deploy_complete_solution' method."
            
            # Get configuration from secrets or environment variables
            try:
                if hasattr(st, 'secrets') and st.secrets is not None:
                    subscription_id = (st.secrets.get('AZURE_SUBSCRIPTION_ID') or os.getenv('AZURE_SUBSCRIPTION_ID', '') or '').strip()
                    resource_group = (st.secrets.get('AZURE_RESOURCE_GROUP') or os.getenv('AZURE_RESOURCE_GROUP', '') or '').strip()
                    factory_name = (st.secrets.get('AZURE_DATA_FACTORY') or os.getenv('AZURE_DATA_FACTORY', '') or '').strip()
                    location = (st.secrets.get('AZURE_LOCATION') or os.getenv('AZURE_LOCATION', 'East US') or 'East US').strip()
                    tenant_id = (st.secrets.get('AZURE_TENANT_ID') or os.getenv('AZURE_TENANT_ID', '') or '').strip()
                    client_id = (st.secrets.get('AZURE_CLIENT_ID') or os.getenv('AZURE_CLIENT_ID', '') or '').strip()
                    client_secret = (st.secrets.get('AZURE_CLIENT_SECRET') or os.getenv('AZURE_CLIENT_SECRET', '') or '').strip()
                else:
                    subscription_id = (os.getenv('AZURE_SUBSCRIPTION_ID', '') or '').strip()
                    resource_group = (os.getenv('AZURE_RESOURCE_GROUP', '') or '').strip()
                    factory_name = (os.getenv('AZURE_DATA_FACTORY', '') or '').strip()
                    location = (os.getenv('AZURE_LOCATION', 'East US') or 'East US').strip()
                    tenant_id = (os.getenv('AZURE_TENANT_ID', '') or '').strip()
                    client_id = (os.getenv('AZURE_CLIENT_ID', '') or '').strip()
                    client_secret = (os.getenv('AZURE_CLIENT_SECRET', '') or '').strip()
            except Exception as e:
                subscription_id = (os.getenv('AZURE_SUBSCRIPTION_ID', '') or '').strip()
                resource_group = (os.getenv('AZURE_RESOURCE_GROUP', '') or '').strip()
                factory_name = (os.getenv('AZURE_DATA_FACTORY', '') or '').strip()
                location = (os.getenv('AZURE_LOCATION', 'East US') or 'East US').strip()
                tenant_id = (os.getenv('AZURE_TENANT_ID', '') or '').strip()
                client_id = (os.getenv('AZURE_CLIENT_ID', '') or '').strip()
                client_secret = (os.getenv('AZURE_CLIENT_SECRET', '') or '').strip()
            
            # Validate required credentials - check for empty strings and treat as missing
            missing_vars = []
            if not tenant_id:
                missing_vars.append('AZURE_TENANT_ID')
            if not client_id:
                missing_vars.append('AZURE_CLIENT_ID')
            if not client_secret:
                missing_vars.append('AZURE_CLIENT_SECRET')
            if not subscription_id:
                missing_vars.append('AZURE_SUBSCRIPTION_ID')
            if not resource_group:
                missing_vars.append('AZURE_RESOURCE_GROUP')
            if not factory_name:
                missing_vars.append('AZURE_DATA_FACTORY')
            
            if missing_vars:
                return False, (
                    f"Azure credentials not configured. Missing or empty environment variables: {', '.join(missing_vars)}. "
                    f"Please set these in Azure Web App → Configuration → Application Settings. "
                    f"Current status: tenant_id={'SET' if tenant_id else 'MISSING'}, "
                    f"client_id={'SET' if client_id else 'MISSING'}, "
                    f"client_secret={'SET' if client_secret else 'MISSING'}"
                )
            
            # Credentials are validated - use them directly (don't normalize to None)
            # Check the class __init__ signature to see what parameters it accepts
            import inspect
            try:
                init_sig = inspect.signature(pipeline_class.__init__)
                init_params = list(init_sig.parameters.keys())[1:]  # Skip 'self'
                
                # Build kwargs dictionary with only accepted parameters
                init_kwargs = {}
                if 'subscription_id' in init_params:
                    init_kwargs['subscription_id'] = subscription_id
                if 'resource_group' in init_params:
                    init_kwargs['resource_group'] = resource_group
                if 'factory_name' in init_params:
                    init_kwargs['factory_name'] = factory_name
                if 'location' in init_params:
                    init_kwargs['location'] = location
                if 'use_timestamp' in init_params:
                    init_kwargs['use_timestamp'] = False
                
                # CRITICAL: If class accepts credentials, ensure ALL three are passed
                accepts_credentials = any(param in init_params for param in ['tenant_id', 'client_id', 'client_secret'])
                
                if accepts_credentials:
                    # Validate each credential individually before adding
                    if 'tenant_id' in init_params:
                        if not tenant_id:
                            return False, "AZURE_TENANT_ID is required but missing or empty"
                        init_kwargs['tenant_id'] = tenant_id
                    
                    if 'client_id' in init_params:
                        if not client_id:
                            return False, "AZURE_CLIENT_ID is required but missing or empty"
                        init_kwargs['client_id'] = client_id
                    
                    if 'client_secret' in init_params:
                        if not client_secret:
                            return False, "AZURE_CLIENT_SECRET is required but missing or empty"
                        init_kwargs['client_secret'] = client_secret
                    
                    # Final validation: ensure all required credentials are present
                    required_creds = []
                    if 'tenant_id' in init_params and not init_kwargs.get('tenant_id'):
                        required_creds.append('tenant_id')
                    if 'client_id' in init_params and not init_kwargs.get('client_id'):
                        required_creds.append('client_id')
                    if 'client_secret' in init_params and not init_kwargs.get('client_secret'):
                        required_creds.append('client_secret')
                    
                    if required_creds:
                        return False, f"Missing required credentials: {', '.join(required_creds)}"
                
                # Instantiate the pipeline class
                try:
                    pipeline_instance = pipeline_class(**init_kwargs)
                except ValueError as cred_error:
                    # Check if it's a credential-related error
                    error_msg = str(cred_error).lower()
                    if 'credential' in error_msg or 'tenant_id' in error_msg or 'client_id' in error_msg:
                        # Provide detailed error with credential status
                        cred_status = {
                            'tenant_id': 'SET' if tenant_id else 'MISSING',
                            'client_id': 'SET' if client_id else 'MISSING',
                            'client_secret': 'SET' if client_secret else 'MISSING'
                        }
                        return False, (
                            f"Credential error: {str(cred_error)}. "
                            f"Credential status: {cred_status}. "
                            f"Please verify AZURE_TENANT_ID, AZURE_CLIENT_ID, and AZURE_CLIENT_SECRET "
                            f"are set correctly in Azure Web App environment variables."
                        )
                    raise
            except ValueError as cred_error:
                # Catch credential errors specifically (this handles errors from the outer try block)
                error_msg = str(cred_error).lower()
                if 'credential' in error_msg or 'tenant_id' in error_msg or 'client_id' in error_msg:
                    cred_status = {
                        'tenant_id': 'SET' if tenant_id else 'MISSING',
                        'client_id': 'SET' if client_id else 'MISSING',
                        'client_secret': 'SET' if client_secret else 'MISSING'
                    }
                    return False, (
                        f"Credential error when instantiating pipeline: {str(cred_error)}. "
                        f"Credential status: {cred_status}. "
                        f"Please verify AZURE_TENANT_ID, AZURE_CLIENT_ID, and AZURE_CLIENT_SECRET are set in Azure Web App environment variables."
                    )
                raise
            except Exception as inspect_error:
                # Fallback: try with all parameters if signature inspection fails
                try:
                    # Double-check credentials before fallback
                    if not tenant_id or not client_id or not client_secret:
                        return False, (
                            f"Cannot instantiate pipeline: credentials missing. "
                            f"tenant_id={'SET' if tenant_id else 'MISSING'}, "
                            f"client_id={'SET' if client_id else 'MISSING'}, "
                            f"client_secret={'SET' if client_secret else 'MISSING'}"
                        )
                    
                    pipeline_instance = pipeline_class(
                        subscription_id=subscription_id,
                        resource_group=resource_group,
                        factory_name=factory_name,
                        location=location,
                        use_timestamp=False,
                        tenant_id=tenant_id,
                        client_id=client_id,
                        client_secret=client_secret
                    )
                except ValueError as cred_error2:
                    if 'credential' in str(cred_error2).lower():
                        return False, (
                            f"Credential error in fallback: {str(cred_error2)}. "
                            f"Credentials passed: tenant_id={'SET' if tenant_id else 'MISSING'}, "
                            f"client_id={'SET' if client_id else 'MISSING'}, "
                            f"client_secret={'SET' if client_secret else 'MISSING'}. "
                            f"Please verify environment variables in Azure Web App."
                        )
                    raise
                except TypeError as type_error:
                    # If that fails, try with minimal parameters
                    try:
                        pipeline_instance = pipeline_class(
                            subscription_id=subscription_id,
                            resource_group=resource_group,
                            factory_name=factory_name
                        )
                    except:
                        return False, f"Failed to instantiate pipeline class. Error: {str(type_error)}. Signature inspection error: {str(inspect_error)}"
            
            # Check if deploy_complete_solution requires parameters
            import inspect
            sig = inspect.signature(pipeline_instance.deploy_complete_solution)
            params = list(sig.parameters.keys())
            
            # Call deploy_complete_solution
            if len(params) > 1:  # Has additional parameters besides self
                # Try to get SQL and blob config if needed
                try:
                    # Try to get from Streamlit secrets first
                    if hasattr(st, 'secrets') and st.secrets is not None:
                        sql_config = {
                            'server_name': (st.secrets.get('AZURE_SQL_SERVER') or os.getenv('AZURE_SQL_SERVER', '') or '').strip(),
                            'database_name': (st.secrets.get('AZURE_SQL_DATABASE') or os.getenv('AZURE_SQL_DATABASE', '') or '').strip(),
                            'username': (st.secrets.get('AZURE_SQL_USER') or os.getenv('AZURE_SQL_USER', '') or '').strip(),
                            'password': (st.secrets.get('AZURE_SQL_PASSWORD') or os.getenv('AZURE_SQL_PASSWORD', '') or '').strip()
                        }
                        blob_config = {
                            'account_name': (st.secrets.get('AZURE_STORAGE_ACCOUNT') or os.getenv('AZURE_STORAGE_ACCOUNT', '') or '').strip(),
                            'account_key': (st.secrets.get('AZURE_STORAGE_KEY') or os.getenv('AZURE_STORAGE_KEY', '') or '').strip()
                        }
                    else:
                        sql_config = {
                            'server_name': (os.getenv('AZURE_SQL_SERVER', '') or '').strip(),
                            'database_name': (os.getenv('AZURE_SQL_DATABASE', '') or '').strip(),
                            'username': (os.getenv('AZURE_SQL_USER', '') or '').strip(),
                            'password': (os.getenv('AZURE_SQL_PASSWORD', '') or '').strip()
                        }
                        blob_config = {
                            'account_name': (os.getenv('AZURE_STORAGE_ACCOUNT', '') or '').strip(),
                            'account_key': (os.getenv('AZURE_STORAGE_KEY', '') or '').strip()
                        }
                    
                    # Validate SQL and blob configs are not empty
                    sql_missing = [k for k, v in sql_config.items() if not v]
                    blob_missing = [k for k, v in blob_config.items() if not v]
                    
                    if sql_missing or blob_missing:
                        missing = []
                        if sql_missing:
                            missing.extend([f"AZURE_SQL_{k.upper().replace('_NAME', '')}" for k in sql_missing])
                        if blob_missing:
                            missing.extend([f"AZURE_STORAGE_{k.upper().replace('_NAME', '').replace('_KEY', '_KEY')}" for k in blob_missing])
                        return False, (
                            f"Missing configuration for deployment: {', '.join(missing)}. "
                            f"Please set these in Azure Web App → Configuration → Application Settings."
                        )
                    
                    pipeline_instance.deploy_complete_solution(sql_config, blob_config)
                except Exception as e:
                    # If it fails, try without parameters (for classes that don't need them)
                    try:
                        pipeline_instance.deploy_complete_solution()
                    except Exception as e2:
                        return False, f"Deployment failed: {str(e2)}. Original error: {str(e)}"
            else:
                pipeline_instance.deploy_complete_solution()
            
            st.session_state.pipeline_instance = pipeline_instance
            if hasattr(pipeline_instance, 'names'):
                st.session_state.pipeline_names = pipeline_instance.names
            
            return True, "Successfully deployed all the resources without any issues"
        except TypeError as e:
            # Specific handling for missing required arguments
            error_msg = str(e)
            if "missing" in error_msg and "required" in error_msg:
                # Extract the class and missing argument from the error
                import re
                match = re.search(r"(\w+)\.init\(\) missing.*?argument: ['\"]?(\w+)['\"]?", error_msg)
                if match:
                    class_name = match.group(1)
                    missing_arg = match.group(2)
                    
                    # Special handling for AzureMLExecutePipelineActivity - this is likely a code generation error
                    if "AzureMLExecutePipelineActivity" in class_name:
                        # Try to auto-fix by searching and replacing in generated code
                        fixed_code = st.session_state.get('generated_code', '')
                        if fixed_code:
                            # Pattern to find AzureMLExecutePipelineActivity instantiations without name
                            # Try to replace with ExecuteDataFlowActivity or add name parameter
                            patterns_to_fix = [
                                (r'AzureMLExecutePipelineActivity\s*\(', 'ExecuteDataFlowActivity('),
                            ]
                            
                            for pattern, replacement in patterns_to_fix:
                                if re.search(pattern, fixed_code):
                                    try:
                                        # Try to replace all instances
                                        auto_fixed = re.sub(pattern, replacement, fixed_code)
                                        st.session_state.generated_code = auto_fixed
                                        st.session_state.code_auto_fixed = True
                                        # Return message asking to retry
                                        return False, f"Auto-fixed: Replaced '{class_name}' with 'ExecuteDataFlowActivity'. Please click 'Execute Code' again to deploy."
                                    except Exception as fix_error:
                                        pass
                        
                        return False, f"Code generation issue: '{class_name}' should not be used here. It's missing required argument '{missing_arg}'. The code likely should use 'ExecuteDataFlowActivity' instead. Please regenerate the code in Tab 2 or manually replace 'AzureMLExecutePipelineActivity' with 'ExecuteDataFlowActivity' in the generated code."
                    
                    return False, f"Code generation issue: {class_name} is missing required argument '{missing_arg}'. Please regenerate the code or fix it manually."
            import traceback
            error_msg_full = traceback.format_exc()
            print(f"TypeError deploying code: {error_msg_full}")
            return False, f"Deployment failed (TypeError): {error_msg}"
        except Exception as e:
            import traceback
            error_msg = traceback.format_exc()
            print(f"Error deploying code: {error_msg}")
            # Provide more helpful error message
            error_type = type(e).__name__
            error_str = str(e)
            if "AzureMLExecutePipelineActivity" in error_str:
                return False, f"Code generation issue: The generated code has an error with AzureMLExecutePipelineActivity. It's missing a required 'name' parameter. Please regenerate the code in Tab 2 or fix it manually in the code editor."
            return False, f"Deployment failed ({error_type}): {error_str}"
    
    def get_pipeline_name_from_code(self, generated_code):
        """Extract pipeline name from generated code"""
        # First, try to get from stored pipeline names if available
        if 'pipeline_names' in st.session_state:
            pipeline_names = st.session_state.pipeline_names
            if isinstance(pipeline_names, dict) and 'pipeline' in pipeline_names:
                return pipeline_names['pipeline']
        
        import re
        pipeline_name = None
        
        # Method 1: Extract from create_pipeline method specifically (MOST RELIABLE)
        # Look for name = '...Pipeline...' inside create_pipeline method
        pipeline_method_match = re.search(
            r"def create_pipeline\(self\):.*?name\s*=\s*['\"]([^'\"]*Pipeline[^'\"]*)['\"]",
            generated_code,
            re.DOTALL
        )
        if pipeline_method_match:
            pipeline_name = pipeline_method_match.group(1)
            # Validate it contains "Pipeline"
            if 'Pipeline' in pipeline_name:
                return pipeline_name
        
        # Method 2: If not found, search within create_pipeline method context
        if not pipeline_name:
            create_pipeline_start = generated_code.find("def create_pipeline(self):")
            if create_pipeline_start != -1:
                # Get method content (next 500 chars should contain the name)
                method_content = generated_code[create_pipeline_start:create_pipeline_start + 500]
                # Look for name = '...Pipeline...' pattern
                name_match = re.search(r"name\s*=\s*['\"]([^'\"]*Pipeline[^'\"]*)['\"]", method_content)
                if name_match:
                    pipeline_name = name_match.group(1)
                    if 'Pipeline' in pipeline_name:
                        return pipeline_name
        
        # Method 3: Try to find patterns like: names['pipeline'] = 'PipelineName'
        patterns = [
            r"names\['pipeline'\]\s*=\s*['\"]([^'\"]*Pipeline[^'\"]*)['\"]",  # names['pipeline'] = 'PipelineName'
            r"['\"]pipeline['\"]:\s*['\"]([^'\"]*Pipeline[^'\"]*)['\"]",  # names['pipeline']: 'PipelineName'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, generated_code)
            if matches:
                # Filter to only names containing "Pipeline"
                valid_matches = [m for m in matches if 'Pipeline' in m]
                if valid_matches:
                    return valid_matches[-1]  # Return the last valid match
        
        # Method 4: Extract from class name and construct pipeline name
        class_match = re.search(r"class\s+(\w+CSVToSQLPipeline)", generated_code)
        if class_match:
            return class_match.group(1)
        
        # Method 5: Try generic class name pattern
        class_match = re.search(r"class\s+(\w+.*?Pipeline)", generated_code)
        if class_match:
            class_name = class_match.group(1)
            # If class name already contains Pipeline, use it directly
            if 'Pipeline' in class_name:
                return class_name
        
        return None
    
    def run_pipeline(self, resource_group, factory_name, pipeline_name, parameters=None):
        """Execute a pipeline in Azure Data Factory"""
        try:
            # Validate pipeline name is not None or empty
            if not pipeline_name:
                return None, "Pipeline name is empty or None"
            
            # Validate pipeline name looks like a pipeline name (contains "Pipeline")
            if 'Pipeline' not in pipeline_name:
                return None, f"Invalid pipeline name format: '{pipeline_name}'. Expected name containing 'Pipeline'."
            
            # Validate pipeline name is not a linked service or dataset name
            if pipeline_name in ['SQLLinkedService', 'BlobStorageLinkedService']:
                return None, f"ERROR: Pipeline name cannot be a linked service name: '{pipeline_name}'. This indicates incorrect name extraction."
            
            adf_client = self.get_adf_client()
            if adf_client is None:
                return None, "Failed to create ADF client"
            
            print(f"Attempting to run pipeline: {pipeline_name} in factory: {factory_name}")
            run_response = adf_client.pipelines.create_run(
                resource_group,
                factory_name,
                pipeline_name,
                parameters=parameters or {}
            )
            
            return run_response.run_id, "Pipeline started successfully"
        except Exception as e:
            error_msg = str(e)
            print(f"Error running pipeline '{pipeline_name}': {error_msg}")
            # Provide more helpful error message
            if "not found" in error_msg.lower():
                return None, f"Failed to start pipeline: Pipeline '{pipeline_name}' not found in factory '{factory_name}'. Make sure deployment completed successfully."
            return None, f"Failed to start pipeline: {error_msg}"
    
    def get_pipeline_status(self, resource_group, factory_name, run_id):
        """Get the status of a pipeline run"""
        try:
            adf_client = self.get_adf_client()
            if adf_client is None:
                return None, "Failed to create ADF client"
            
            pipeline_run = adf_client.pipeline_runs.get(
                resource_group,
                factory_name,
                run_id
            )
            
            status = pipeline_run.status
            message = ""
            
            if hasattr(pipeline_run, 'message') and pipeline_run.message:
                message = pipeline_run.message
            
            return status, message
        except Exception as e:
            print(f"Error getting pipeline status: {e}")
            return None, f"Error retrieving status: {str(e)}"