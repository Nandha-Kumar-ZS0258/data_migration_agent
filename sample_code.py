
import os
import time
from datetime import datetime
from azure.identity import ClientSecretCredential
from azure.mgmt.datafactory import DataFactoryManagementClient
from azure.mgmt.datafactory.models import *


class SalesCSVToSQLPipeline:
    """Creates CSV to SQL data pipeline with fact/dimension splitting"""

    def __init__(self, subscription_id, resource_group, factory_name, location='eastus', 
                 use_timestamp=False, tenant_id=None, client_id=None, client_secret=None):
        self.subscription_id = subscription_id
        self.resource_group = resource_group
        self.factory_name = factory_name
        self.location = location
        self.use_timestamp = use_timestamp

        # Store credentials
        self.tenant_id = tenant_id
        self.client_id = client_id
        self.client_secret = client_secret

        # Generate timestamp for resource naming
        self.timestamp = datetime.now().strftime('%Y%m%d%H%M%S')

        # Define resource names
        self.names = self.generate_resource_names()

        # Authenticate
        self.credential = self.get_credential()
        self.client = DataFactoryManagementClient(self.credential, subscription_id)

    def generate_resource_names(self):
        """Generate resource names with optional timestamps"""
        suffix = f"_{self.timestamp}" if self.use_timestamp else ""

        return {
            # Linked Services
            'sql_linked_service': f'SQLLinkedService{suffix}',
            'blob_linked_service': f'BlobStorageLinkedService{suffix}',

            # Datasets
            'source_csv_dataset': f'SourceSalesCSVDataset{suffix}',
            'fact_sales_dataset': f'FactSalesDataset{suffix}',
            'dim_product_dataset': f'DimProductDataset{suffix}',
            'dim_customer_dataset': f'DimCustomerDataset{suffix}',
            'dim_time_dataset': f'DimTimeDataset{suffix}',

            # Data Flows
            'dimension_dataflow': f'LoadDimensionsDataFlow{suffix}',
            'fact_dataflow': f'LoadFactDataFlow{suffix}',

            # Pipeline
            'pipeline': f'SalesCSVToSQLPipeline{suffix}'
        }

    def get_credential(self):
        """Get Azure credential from instance variables"""
        if all([self.tenant_id, self.client_id, self.client_secret]):
            print(f"Using Service Principal authentication")
            return ClientSecretCredential(
                tenant_id=self.tenant_id,
                client_id=self.client_id,
                client_secret=self.client_secret
            )

        raise ValueError(
            "Azure credentials not provided. Pass tenant_id, client_id, and client_secret "
            "to the SalesCSVToSQLPipeline constructor."
        )

    # ==================== Linked Services ====================

    def create_sql_linked_service(self, server_name, database_name, username, password):
        """Create SQL Linked Service"""
        name = self.names['sql_linked_service']
        print(f"Creating SQL Linked Service: {name}...")

        connection_string = (
            f"Server=tcp:{server_name},1433;"
            f"Database={database_name};"
            f"User ID={username};"
            f"Password={password};"
            "Encrypt=True;Connection Timeout=30;"
        )

        properties = AzureSqlDatabaseLinkedService(
            connection_string=SecureString(value=connection_string)
        )

        linked_service = LinkedServiceResource(properties=properties)

        result = self.client.linked_services.create_or_update(
            self.resource_group,
            self.factory_name,
            name,
            linked_service
        )
        print(f"✓ SQL Linked Service created: {result.name}")
        return result

    def create_blob_storage_linked_service(self, account_name, account_key):
        """Create Azure Blob Storage Linked Service"""
        name = self.names['blob_linked_service']
        print(f"Creating Blob Storage Linked Service: {name}...")

        connection_string = (
            f"DefaultEndpointsProtocol=https;"
            f"AccountName={account_name};"
            f"AccountKey={account_key};"
            "EndpointSuffix=core.windows.net"
        )

        properties = AzureBlobStorageLinkedService(
            connection_string=SecureString(value=connection_string)
        )

        linked_service = LinkedServiceResource(properties=properties)

        result = self.client.linked_services.create_or_update(
            self.resource_group,
            self.factory_name,
            name,
            linked_service
        )
        print(f"✓ Blob Storage Linked Service created: {result.name}")
        return result

    # ==================== Datasets ====================

    def create_source_csv_dataset(self, container_name='applicationdata', folder_path='source', file_name='sales_data_sample.csv'):
        """Create source CSV dataset"""
        name = self.names['source_csv_dataset']
        print(f"Creating Source CSV Dataset: {name}...")
        print(f"  Container: {container_name}")
        print(f"  Directory: {folder_path}")
        print(f"  File: {file_name}")

        properties = DelimitedTextDataset(
            linked_service_name=LinkedServiceReference(
                reference_name=self.names['blob_linked_service'],
                type='LinkedServiceReference'
            ),
            location=AzureBlobStorageLocation(
                container=container_name,
                folder_path=folder_path,
                file_name=file_name
            ),
            column_delimiter=',',
            encoding_name='UTF-8',
            first_row_as_header=True
        )

        dataset = DatasetResource(properties=properties)

        result = self.client.datasets.create_or_update(
            self.resource_group,
            self.factory_name,
            name,
            dataset
        )
        print(f"✓ Source CSV Dataset created: {result.name}")
        return result

    def create_sql_datasets(self):
        """Create all SQL table datasets with explicit dbo schema"""
        tables = [
            ('fact_sales_dataset', 'dbo', 'FactSales'),
            ('dim_product_dataset', 'dbo', 'DimProduct'),
            ('dim_customer_dataset', 'dbo', 'DimCustomer'),
            ('dim_time_dataset', 'dbo', 'DimTime')
        ]

        results = []
        for dataset_key, schema_name, table_name in tables:
            name = self.names[dataset_key]
            print(f"Creating {schema_name}.{table_name} Dataset: {name}...")

            properties = AzureSqlTableDataset(
                linked_service_name=LinkedServiceReference(
                    reference_name=self.names['sql_linked_service'],
                    type='LinkedServiceReference'
                ),
                schema=schema_name,
                table=table_name
            )

            dataset = DatasetResource(properties=properties)

            result = self.client.datasets.create_or_update(
                self.resource_group,
                self.factory_name,
                name,
                dataset
            )
            print(f"✓ {table_name} Dataset created: {result.name}")
            results.append(result)

        return results

    # ==================== Data Flows ====================

    def create_dimension_dataflow(self):
        """
        Create data flow to load dimension tables only
        This runs FIRST to populate reference data
        """
        name = self.names['dimension_dataflow']
        print(f"Creating Dimension Data Flow: {name}...")

        script = """
source(output(
      PRODUCTCODE as string,
      PRODUCTLINE as string,
      MSRP as string,
      CUSTOMERNAME as string,
      PHONE as string,
      ADDRESSLINE1 as string,
      ADDRESSLINE2 as string,
      CITY as string,
      STATE as string,
      POSTALCODE as string,
      COUNTRY as string,
      TERRITORY as string,
      CONTACTLASTNAME as string,
      CONTACTFIRSTNAME as string,
      ORDERDATE as string,
      QTR_ID as string,
      MONTH_ID as string,
      YEAR_ID as string
 ),
 allowSchemaDrift: true,
 validateSchema: false,
 ignoreNoFilesFound: false) ~> SourceCSV

SourceCSV select(mapColumn(
      PRODUCTCODE,
      PRODUCTLINE,
      MSRP
 ),
 skipDuplicateMapInputs: true,
 skipDuplicateMapOutputs: true) ~> SelectDimProduct

SelectDimProduct aggregate(groupBy(PRODUCTCODE),
 PRODUCTLINE = first(PRODUCTLINE),
     MSRP = first(MSRP)) ~> AggregateDimProduct

AggregateDimProduct cast(output(
      MSRP as decimal(10,2)
 ),
 errors: true) ~> CastDimProduct

CastDimProduct sink(allowSchemaDrift: true,
 validateSchema: false,
 deletable:false,
 insertable:true,
 updateable:false,
 upsertable:false,
 format: 'table',
 skipDuplicateMapInputs: true,
 skipDuplicateMapOutputs: true,
 errorHandlingOption: 'stopOnFirstError') ~> LoadDimProduct

SourceCSV select(mapColumn(
      CUSTOMERNAME,
      PHONE,
      ADDRESSLINE1,
      ADDRESSLINE2,
      CITY,
      STATE,
      POSTALCODE,
      COUNTRY,
      TERRITORY,
      CONTACTLASTNAME,
      CONTACTFIRSTNAME
 ),
 skipDuplicateMapInputs: true,
 skipDuplicateMapOutputs: true) ~> SelectDimCustomer

SelectDimCustomer aggregate(groupBy(CUSTOMERNAME),
 PHONE = first(PHONE),
     ADDRESSLINE1 = first(ADDRESSLINE1),
     ADDRESSLINE2 = first(ADDRESSLINE2),
     CITY = first(CITY),
     STATE = first(STATE),
     POSTALCODE = first(POSTALCODE),
     COUNTRY = first(COUNTRY),
     TERRITORY = first(TERRITORY),
     CONTACTLASTNAME = first(CONTACTLASTNAME),
     CONTACTFIRSTNAME = first(CONTACTFIRSTNAME)) ~> AggregateDimCustomer

AggregateDimCustomer sink(allowSchemaDrift: true,
 validateSchema: false,
 deletable:false,
 insertable:true,
 updateable:false,
 upsertable:false,
 format: 'table',
 skipDuplicateMapInputs: true,
 skipDuplicateMapOutputs: true,
 errorHandlingOption: 'stopOnFirstError') ~> LoadDimCustomer

SourceCSV select(mapColumn(
      ORDERDATE,
      QTR_ID,
      MONTH_ID,
      YEAR_ID
 ),
 skipDuplicateMapInputs: true,
 skipDuplicateMapOutputs: true) ~> SelectDimTime

SelectDimTime aggregate(groupBy(ORDERDATE),
 QTR_ID = first(QTR_ID),
 MONTH_ID = first(MONTH_ID),
 YEAR_ID = first(YEAR_ID)) ~> AggregateDimTime

AggregateDimTime derive(ORDERDATE = toDate(ORDERDATE, 'M/d/yyyy')) ~> DeriveDimTime

DeriveDimTime cast(output(
      QTR_ID as integer,
      MONTH_ID as integer,
      YEAR_ID as integer
 ),
 errors: true) ~> CastDimTime

CastDimTime sink(allowSchemaDrift: true,
 validateSchema: false,
 deletable:false,
 insertable:true,
 updateable:false,
 upsertable:false,
 format: 'table',
 skipDuplicateMapInputs: true,
 skipDuplicateMapOutputs: true,
 errorHandlingOption: 'stopOnFirstError') ~> LoadDimTime
"""
        dataflow_properties = MappingDataFlow(
            sources=[
                DataFlowSource(
                    name='SourceCSV',
                    dataset=DatasetReference(
                        reference_name=self.names['source_csv_dataset'],
                        type='DatasetReference'
                    )
                )
            ],
            sinks=[
                DataFlowSink(
                    name='LoadDimProduct',
                    dataset=DatasetReference(
                        reference_name=self.names['dim_product_dataset'],
                        type='DatasetReference'
                    )
                ),
                DataFlowSink(
                    name='LoadDimCustomer',
                    dataset=DatasetReference(
                        reference_name=self.names['dim_customer_dataset'],
                        type='DatasetReference'
                    )
                ),
                DataFlowSink(
                    name='LoadDimTime',
                    dataset=DatasetReference(
                        reference_name=self.names['dim_time_dataset'],
                        type='DatasetReference'
                    )
                )
            ],
            transformations=[
                Transformation(name='SelectDimProduct'),
                Transformation(name='AggregateDimProduct'),
                Transformation(name='CastDimProduct'),
                Transformation(name='SelectDimCustomer'),
                Transformation(name='AggregateDimCustomer'),
                Transformation(name='SelectDimTime'),
                Transformation(name='AggregateDimTime'),
                Transformation(name='DeriveDimTime'),
                Transformation(name='CastDimTime')
            ],
            script=script
        )

        dataflow = DataFlowResource(properties=dataflow_properties)

        result = self.client.data_flows.create_or_update(
            self.resource_group,
            self.factory_name,
            name,
            dataflow
        )
        print(f"✓ Dimension Data Flow created: {result.name}")
        return result

    def create_fact_dataflow(self):
        """
        Create data flow to load fact table only
        This runs SECOND after dimensions are loaded
        """
        name = self.names['fact_dataflow']
        print(f"Creating Fact Data Flow: {name}...")

        script = """
source(output(
      ORDERNUMBER as string,
      QUANTITYORDERED as string,
      PRICEEACH as string,
      ORDERLINENUMBER as string,
      SALES as string,
      ORDERDATE as string,
      STATUS as string,
      QTR_ID as string,
      MONTH_ID as string,
      YEAR_ID as string,
      PRODUCTCODE as string,
      CUSTOMERNAME as string,
      PHONE as string,
      ADDRESSLINE1 as string,
      ADDRESSLINE2 as string,
      CITY as string,
      STATE as string,
      POSTALCODE as string,
      COUNTRY as string,
      TERRITORY as string,
      CONTACTLASTNAME as string,
      CONTACTFIRSTNAME as string
 ),
 allowSchemaDrift: true,
 validateSchema: false,
 ignoreNoFilesFound: false) ~> SourceCSV

SourceCSV select(mapColumn(
      ORDERNUMBER,
      QUANTITYORDERED,
      PRICEEACH,
      ORDERLINENUMBER,
      SALES,
      ORDERDATE,
      STATUS,
      QTR_ID,
      MONTH_ID,
      YEAR_ID,
      PRODUCTCODE,
      CUSTOMERNAME
 ),
 skipDuplicateMapInputs: true,
 skipDuplicateMapOutputs: true) ~> SelectFactSales

SelectFactSales derive(ORDERDATE = toDate(ORDERDATE, 'M/d/yyyy')) ~> DeriveFactSales

DeriveFactSales cast(output(
      ORDERNUMBER as integer,
      QUANTITYORDERED as integer,
      PRICEEACH as decimal(10,2),
      SALES as decimal(15,2),
      QTR_ID as integer,
      MONTH_ID as integer,
      YEAR_ID as integer
 ),
 errors: true) ~> CastFactSales

CastFactSales sink(allowSchemaDrift: true,
 validateSchema: false,
 deletable:false,
 insertable:true,
 updateable:false,
 upsertable:false,
 format: 'table',
 skipDuplicateMapInputs: true,
 skipDuplicateMapOutputs: true,
 errorHandlingOption: 'stopOnFirstError') ~> LoadFactSales
"""
        dataflow_properties = MappingDataFlow(
            sources=[
                DataFlowSource(
                    name='SourceCSV',
                    dataset=DatasetReference(
                        reference_name=self.names['source_csv_dataset'],
                        type='DatasetReference'
                    )
                )
            ],
            sinks=[
                DataFlowSink(
                    name='LoadFactSales',
                    dataset=DatasetReference(
                        reference_name=self.names['fact_sales_dataset'],
                        type='DatasetReference'
                    )
                )
            ],
            transformations=[
                Transformation(name='SelectFactSales'),
                Transformation(name='DeriveFactSales'),
                Transformation(name='CastFactSales')
            ],
            script=script
        )

        dataflow = DataFlowResource(properties=dataflow_properties)

        result = self.client.data_flows.create_or_update(
            self.resource_group,
            self.factory_name,
            name,
            dataflow
        )
        print(f"✓ Fact Data Flow created: {result.name}")
        return result

    # ==================== Pipeline ====================

    def create_pipeline(self):
        """
        Create main pipeline with TWO dataflow activities in sequence:
        1. Load Dimensions (DimProduct, DimCustomer, DimTime)
        2. Load Fact (FactSales) - depends on dimensions completing
        """
        name = self.names['pipeline']
        print(f"Creating Pipeline: {name}...")

        # Activity 1: Execute Data Flow - Load Dimensions FIRST
        dimension_activity = ExecuteDataFlowActivity(
            name='LoadDimensions',
            policy=ActivityPolicy(
                timeout='0.12:00:00',
                retry=0,
                retry_interval_in_seconds=30,
                secure_output=False,
                secure_input=False
            ),
            data_flow=DataFlowReference(
                reference_name=self.names['dimension_dataflow'],
                type='DataFlowReference'
            ),
            compute=ExecuteDataFlowActivityTypePropertiesCompute(
                compute_type='General',
                core_count=8
            ),
            trace_level='Fine'
        )

        # Activity 2: Execute Data Flow - Load Fact AFTER dimensions
        fact_activity = ExecuteDataFlowActivity(
            name='LoadFactSales',
            depends_on=[
                ActivityDependency(
                    activity='LoadDimensions',
                    dependency_conditions=['Succeeded']
                )
            ],
            policy=ActivityPolicy(
                timeout='0.12:00:00',
                retry=0,
                retry_interval_in_seconds=30,
                secure_output=False,
                secure_input=False
            ),
            data_flow=DataFlowReference(
                reference_name=self.names['fact_dataflow'],
                type='DataFlowReference'
            ),
            compute=ExecuteDataFlowActivityTypePropertiesCompute(
                compute_type='General',
                core_count=8
            ),
            trace_level='Fine'
        )

        # Create pipeline with both activities in sequence
        pipeline = PipelineResource(
            description='Sales CSV to SQL pipeline - dimensions first, then fact table',
            activities=[dimension_activity, fact_activity]
        )

        result = self.client.pipelines.create_or_update(
            self.resource_group,
            self.factory_name,
            name,
            pipeline
        )
        print(f"✓ Pipeline created: {result.name}")
        return result

    # ==================== Deployment ====================

    def deploy_complete_solution(self, sql_config, blob_config):
        """
        Deploy complete Sales CSV to SQL pipeline solution

        Args:
            sql_config: dict with keys: server_name, database_name, username, password
            blob_config: dict with keys: account_name, account_key

        Note: CSV file location is now hardcoded to:
              Container: applicationdata
              Directory: source
              File: sales_data_sample.csv
        """
        print("=" * 80)
        print("DEPLOYING SALES CSV TO SQL PIPELINE")
        print("=" * 80)
        print()

        try:
            # Step 1: Create Linked Services
            print("Step 1: Creating Linked Services")
            print("-" * 80)
            self.create_sql_linked_service(
                sql_config['server_name'],
                sql_config['database_name'],
                sql_config['username'],
                sql_config['password']
            )
            self.create_blob_storage_linked_service(
                blob_config['account_name'],
                blob_config['account_key']
            )
            print()

            # Step 2: Create Datasets
            print("Step 2: Creating Datasets")
            print("-" * 80)
            self.create_source_csv_dataset()
            self.create_sql_datasets()
            print()

            # Step 3: Create Data Flows
            print("Step 3: Creating Data Flows")
            print("-" * 80)
            self.create_dimension_dataflow()
            self.create_fact_dataflow()
            print()

            # Step 4: Create Pipeline
            print("Step 4: Creating Pipeline")
            print("-" * 80)
            self.create_pipeline()
            print()

            print("=" * 80)
            print("✓ DEPLOYMENT COMPLETED SUCCESSFULLY!")
            print("=" * 80)
            print()
            print("Resources Created:")
            print(f"  Pipeline: {self.names['pipeline']}")
            print(f"    ├── Activity 1: LoadDimensions (Data Flow) - RUNS FIRST")
            print(f"    │   └── Data Flow: {self.names['dimension_dataflow']}")
            print(f"    │       ├── Source: applicationdata/source/sales_data_sample.csv")
            print(f"    │       ├── Sink: dbo.DimProduct")
            print(f"    │       ├── Sink: dbo.DimCustomer")
            print(f"    │       └── Sink: dbo.DimTime")
            print(f"    │")
            print(f"    └── Activity 2: LoadFactSales (Data Flow) - RUNS AFTER DIMENSIONS")
            print(f"        └── Data Flow: {self.names['fact_dataflow']}")
            print(f"            ├── Source: applicationdata/source/sales_data_sample.csv")
            print(f"            └── Sink: dbo.FactSales")
            print()
            print("Execution Order:")
            print("  1. Dimensions are loaded first (Product, Customer, Time)")
            print("  2. Fact table is loaded after dimensions succeed")
            print("  3. This prevents foreign key constraint violations")
            print()

        except Exception as e:
            print(f"✗ Deployment failed: {str(e)}")
            import traceback
            traceback.print_exc()
            raise

    # ==================== Pipeline Execution ====================

    def run_pipeline(self, parameters=None):
        """Execute the Sales CSV to SQL pipeline"""
        print("Starting pipeline execution...")

        try:
            run_response = self.client.pipelines.create_run(
                self.resource_group,
                self.factory_name,
                self.names['pipeline'],
                parameters=parameters or {}
            )

            print(f"✓ Pipeline started successfully")
            print(f"  Run ID: {run_response.run_id}")
            return run_response.run_id

        except Exception as e:
            print(f"✗ Failed to start pipeline: {str(e)}")
            return None

    def monitor_pipeline(self, run_id, check_interval=10):
        """Monitor pipeline execution status"""
        if not run_id:
            print("No valid run ID provided")
            return None

        print(f"\nMonitoring pipeline run: {run_id}")
        print("-" * 80)

        try:
            while True:
                pipeline_run = self.client.pipeline_runs.get(
                    self.resource_group,
                    self.factory_name,
                    run_id
                )

                status = pipeline_run.status
                timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
                print(f"[{timestamp}] Status: {status}")

                if status in ['Succeeded', 'Failed', 'Cancelled']:
                    print("-" * 80)
                    if status == 'Succeeded':
                        print("✓ Pipeline execution completed successfully!")
                        if hasattr(pipeline_run, 'duration_in_ms') and pipeline_run.duration_in_ms:
                            print(f"  Duration: {pipeline_run.duration_in_ms / 1000:.2f} seconds")
                    elif status == 'Failed':
                        print("✗ Pipeline execution failed.")
                        print("  Check Azure Portal for detailed error logs.")
                        if hasattr(pipeline_run, 'message') and pipeline_run.message:
                            print(f"  Error: {pipeline_run.message}")
                    else:
                        print("⚠ Pipeline execution was cancelled.")
                    return status

                time.sleep(check_interval)

        except KeyboardInterrupt:
            print("\n⚠ Monitoring interrupted by user")
            return None
        except Exception as e:
            print(f"✗ Error during monitoring: {str(e)}")
            return None


# ==================== Main Execution ====================

def main():
    """
    Main execution function
    Deploy and run the Sales CSV to SQL pipeline
    """


    # ============================================================================
    # CONFIGURATION - Load from environment variables
    # ============================================================================
    SUBSCRIPTION_ID = os.getenv('AZURE_SUBSCRIPTION_ID', 'YOUR_SUBSCRIPTION_ID')
    RESOURCE_GROUP = os.getenv('AZURE_RESOURCE_GROUP', 'YOUR_RESOURCE_GROUP')
    FACTORY_NAME = os.getenv('AZURE_DATA_FACTORY_NAME', 'YOUR_DATA_FACTORY_NAME')
    LOCATION = os.getenv('AZURE_LOCATION', 'eastus')
    TENANT_ID = os.getenv('AZURE_TENANT_ID', 'YOUR_TENANT_ID')
    CLIENT_ID = os.getenv('AZURE_CLIENT_ID', 'YOUR_CLIENT_ID')
    CLIENT_SECRET = os.getenv('AZURE_CLIENT_SECRET', 'YOUR_CLIENT_SECRET')
    
    # SQL Configuration
    sql_config = {
        'server_name': os.getenv('AZURE_SQL_SERVER', 'YOUR_SQL_SERVER'),
        'database_name': os.getenv('AZURE_SQL_DATABASE', 'YOUR_SQL_DATABASE'),
        'username': os.getenv('AZURE_SQL_USERNAME', 'YOUR_SQL_USERNAME'),
        'password': os.getenv('AZURE_SQL_PASSWORD', 'YOUR_SQL_PASSWORD')
    }
    
    # Blob Storage Configuration
    blob_config = {
        'account_name': os.getenv('AZURE_STORAGE_ACCOUNT_NAME', 'YOUR_STORAGE_ACCOUNT_NAME'),
        'account_key': os.getenv('AZURE_STORAGE_ACCOUNT_KEY', 'YOUR_STORAGE_ACCOUNT_KEY')
    }
    
    # ============================================================================

    print("Configuration:")
    print(f"  Subscription ID: {SUBSCRIPTION_ID}")
    print(f"  Resource Group: {RESOURCE_GROUP}")
    print(f"  Data Factory: {FACTORY_NAME}")
    print(f"  Location: {LOCATION}")
    print(f"  CSV Source: applicationdata/source/sales_data_sample.csv")
    print()

    # Initialize pipeline manager with credentials
    pipeline_manager = SalesCSVToSQLPipeline(
        subscription_id=SUBSCRIPTION_ID,
        resource_group=RESOURCE_GROUP,
        factory_name=FACTORY_NAME,
        location=LOCATION,
        use_timestamp=False,
        tenant_id=TENANT_ID,
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET
    )

    # Deploy complete solution
    pipeline_manager.deploy_complete_solution(sql_config, blob_config)

    print("\nDeployment complete!")


if __name__ == '__main__':
    """
    Azure Data Factory - Sales CSV to SQL Pipeline

    This script creates and deploys a complete ADF pipeline that:
    1. Reads a single CSV file from Azure Blob Storage
    2. FIRST loads dimension tables (DimProduct, DimCustomer, DimTime)
    3. THEN loads fact table (FactSales) to prevent FK violations
    4. Transforms data with proper type casting
    5. Loads data into existing Azure SQL Database tables

    KEY UPDATE: Split into TWO dataflows to ensure proper load order:
    - DataFlow 1: Load all dimensions first
    - DataFlow 2: Load fact table after dimensions succeed

    This prevents the FK constraint violation error you encountered.

    Prerequisites:
    - Python packages: azure-identity, azure-mgmt-datafactory
    - CSV file uploaded to blob storage
    - Azure SQL Database with tables already created
    - Service Principal with Data Factory Contributor role

    Usage:
    1. Update configuration in main() function
    2. Run: python sales_csv_to_sql_pipeline.py
    """

    main()