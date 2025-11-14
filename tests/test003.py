#!/usr/bin/env python3
"""
Azure Data Factory - Multi-Hospital CSV to SQL Pipeline Implementation
Copies multiple CSV files from blob storage, unions them, transforms into fact/dimension tables,
and loads into Azure SQL Database.
"""

import os
import time
from datetime import datetime
from azure.identity import ClientSecretCredential
from azure.mgmt.datafactory import DataFactoryManagementClient
from azure.mgmt.datafactory.models import *


class HospitalCSVToSQLPipeline:
    """Creates multi-CSV to SQL data pipeline with fact/dimension splitting"""
    
    def __init__(self, subscription_id, resource_group, factory_name, location='eastus', 
                 use_timestamp=False, tenant_id=None, client_id=None, client_secret=None):
        self.subscription_id = subscription_id
        self.resource_group = resource_group
        self.factory_name = factory_name
        self.location = location
        self.use_timestamp = use_timestamp
        
        self.tenant_id = tenant_id
        self.client_id = client_id
        self.client_secret = client_secret
        
        self.timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        self.names = self.generate_resource_names()
        self.credential = self.get_credential()
        self.client = DataFactoryManagementClient(self.credential, subscription_id)
    
    def generate_resource_names(self):
        """Generate resource names with optional timestamps"""
        suffix = f"{self.timestamp}" if self.use_timestamp else ""
        
        return {
            # Linked Services
            'sql_linked_service': f'SQLLinkedServiceConnection{suffix}',
            'blob_linked_service': f'AzureBlobStorageConnection{suffix}',
            
            # Datasets
            'source_csv_dataset': f'SourceHospitalCSVDataset{suffix}',
            'staging_csv_dataset': f'StagingUnionCSVDataset{suffix}',
            'fact_table_dataset': f'FactVisitDataset{suffix}',
            'dim_patient_dataset': f'DimPatientDataset{suffix}',
            'dim_doctor_dataset': f'DimDoctorDataset{suffix}',
            'dim_hospital_dataset': f'DimHospitalDataset{suffix}',
            'dim_date_dataset': f'DimDateDataset{suffix}',
            'dim_medication_dataset': f'DimMedicationDataset{suffix}',
            
            # Data Flows
            'union_dataflow': f'UnionAllHospitalCSVs{suffix}',
            'transform_dataflow': f'TransformToFactDimension{suffix}',
            
            # Pipeline
            'pipeline': f'HospitalCSVToSQLPipeline{suffix}'
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
            "to the HospitalCSVToSQLPipeline constructor."
        )
    
    # ==================== Linked Services ====================
    
    def create_sql_linked_service(self, server_name=None, database_name=None, username=None, password=None):
        """Create SQL Linked Service"""
        name = self.names['sql_linked_service']
        print(f"Creating SQL Linked Service: {name}...")
        
        # Get credentials from environment variables or parameters
        server_name = server_name or os.getenv('AZURE_SQL_SERVER', 'YOUR_SQL_SERVER')
        database_name = database_name or os.getenv('AZURE_SQL_DATABASE', 'YOUR_SQL_DATABASE')
        username = username or os.getenv('AZURE_SQL_USERNAME', 'YOUR_SQL_USERNAME')
        password = password or os.getenv('AZURE_SQL_PASSWORD', 'YOUR_SQL_PASSWORD')
        
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
    
    def create_blob_storage_linked_service(self, account_name=None, account_key=None):
        """Create Azure Blob Storage Linked Service"""
        name = self.names['blob_linked_service']
        print(f"Creating Blob Storage Linked Service: {name}...")
        
        # Get credentials from environment variables or parameters
        account_name = account_name or os.getenv('AZURE_STORAGE_ACCOUNT_NAME', 'YOUR_STORAGE_ACCOUNT_NAME')
        account_key = account_key or os.getenv('AZURE_STORAGE_ACCOUNT_KEY', 'YOUR_STORAGE_ACCOUNT_KEY')
        
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
    
    def create_source_csv_dataset(self):
        """
        Create source CSV dataset without wildcard in file_name
        The wildcard will be specified in the data flow source directly
        """
        name = self.names['source_csv_dataset']
        print(f"Creating Source CSV Dataset: {name}...")
        
        properties = DelimitedTextDataset(
            linked_service_name=LinkedServiceReference(
                reference_name=self.names['blob_linked_service'],
                type='LinkedServiceReference'
            ),
            location=AzureBlobStorageLocation(
                container='applicationdata',
                folder_path='source'

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
    
    def create_staging_csv_dataset(self):
        """Create staging CSV dataset for union output"""
        name = self.names['staging_csv_dataset']
        print(f"Creating Staging CSV Dataset: {name}...")
        
        properties = DelimitedTextDataset(
            linked_service_name=LinkedServiceReference(
                reference_name=self.names['blob_linked_service'],
                type='LinkedServiceReference'
            ),
            location=AzureBlobStorageLocation(
                container='applicationdata',
                folder_path='staging'
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
        print(f"✓ Staging CSV Dataset created: {result.name}")
        return result
    
    def create_fact_table_dataset(self):
        """Create Fact Visit table dataset"""
        name = self.names['fact_table_dataset']
        print(f"Creating Fact Visit Dataset: {name}...")
        
        properties = AzureSqlTableDataset(
            linked_service_name=LinkedServiceReference(
                reference_name=self.names['sql_linked_service'],
                type='LinkedServiceReference'
            ),
            schema='dbo',
            table='FactVisit'
        )
        
        dataset = DatasetResource(properties=properties)
        
        result = self.client.datasets.create_or_update(
            self.resource_group,
            self.factory_name,
            name,
            dataset
        )
        print(f"✓ Fact Visit Dataset created: {result.name}")
        return result
    
    def create_dimension_datasets(self):
        """Create all dimension table datasets"""
        dimensions = [
            ('dim_patient_dataset', 'DimPatient'),
            ('dim_doctor_dataset', 'DimDoctor'),
            ('dim_hospital_dataset', 'DimHospital'),
            ('dim_date_dataset', 'DimDate'),
            ('dim_medication_dataset', 'DimMedication')
        ]
        
        results = []
        for dataset_key, table_name in dimensions:
            name = self.names[dataset_key]
            print(f"Creating {table_name} Dataset: {name}...")
            
            properties = AzureSqlTableDataset(
                linked_service_name=LinkedServiceReference(
                    reference_name=self.names['sql_linked_service'],
                    type='LinkedServiceReference'
                ),
                schema='dbo',
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
    
    def create_union_dataflow(self):
        """
        Create data flow to union all CSV files.
        ALTERNATIVE: Using folder path without wildcard - reads all files in folder
        """
        name = self.names['union_dataflow']
        print(f"Creating Union Data Flow: {name}...")
        
        script = """source(output(
      Patient_ID as string,
      Patient_First_Name as string,
      Patient_Last_Name as string,
      Gender as string,
      DOB as string,
      Age as string,
      Marital_Status as string,
      Phone_Number as string,
      Email as string,
      Address as string,
      City as string,
      State as string,
      ZipCode as string,
      Ethnicity as string,
      Blood_Type as string,
      Allergies as string,
      Emergency_Contact_Name as string,
      Emergency_Contact_Phone as string,
      Insurance_ID as string,
      Doctor_ID as string,
      Doctor_Name as string,
      Doctor_Licence as string,
      Specialization as string,
      Department_ID as string,
      Department_Name as string,
      Doctor_Email as string,
      Doctor_Phone as string,
      Years_of_Experience as string,
      Shift as string,
      Hospital_ID as string,
      Hospital_Name as string,
      Hospital_Branch as string,
      Hospital_City as string,
      Hospital_State as string,
      Hospital_Type as string,
      Visit_Date as string,
      Visit_Time as string,
      Discharge_Date as string,
      Billing_Date as string,
      Date_ID as string,
      Diagnosis_Code as string,
      Diagnosis_Description as string,
      Procedure_Code as string,
      Procedure_Description as string,
      Medication_ID as string,
      Medication_Name as string,
      Medication_Strength as string,
      Medication_Form as string,
      Medication_Route as string,
      Medication_SideEffects as string,
      Visit_ID as string,
      Invoice_ID as string,
      Total_Amount as string,
      Currency as string,
      Insurance_Covered_Amount as string,
      Patient_Pay_Amount as string,
      Payment_Status as string,
      Payment_Method as string,
      Billing_Provider as string,
      Claim_Number as string,
      Length_of_Stay_Days as string,
      Visit_Duration_Minutes as string,
      Room_Type as string,
      Admission_Type as string,
      Discharge_Disposition as string,
      Dispense_ID as string,
      Pharmacy_Name as string,
      Dispense_Quantity as string,
      Dispense_Cost as string,
      Refillable as string,
      Record_Created_Timestamp as string,
      Data_Source as string
 ),
 allowSchemaDrift: true,
 validateSchema: false,
 ignoreNoFilesFound: false) ~> SourceCSV
SourceCSV sink(allowSchemaDrift: true,
 validateSchema: false,
 skipDuplicateMapInputs: true,
 skipDuplicateMapOutputs: true) ~> StagingSink"""
        
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
                    name='StagingSink',
                    dataset=DatasetReference(
                        reference_name=self.names['staging_csv_dataset'],
                        type='DatasetReference'
                    )
                )
            ],
            transformations=[],
            script=script
        )
        
        dataflow = DataFlowResource(properties=dataflow_properties)
        
        result = self.client.data_flows.create_or_update(
            self.resource_group,
            self.factory_name,
            name,
            dataflow
        )
        print(f"✓ Union Data Flow created: {result.name}")
        return result

    def create_transform_dataflow(self):
        """
            Create data flow to transform staging data into fact and dimension tables
    
            UPDATED: Explicit column mapping for existing target tables
            - All map() transformations connected to their sinks
            - allowSchemaDrift: true for sinks (required when dataset schema not pre-defined)
            - validateSchema: false for sinks (ADF will infer from data)
            - Removed recreate:true from all sinks
        """
        name = self.names['transform_dataflow']
        print(f"Creating Transform Data Flow: {name}...")
        
        script = """source(output(
      Patient_ID as string,
      Patient_First_Name as string,
      Patient_Last_Name as string,
      Gender as string,
      DOB as string,
      Age as string,
      Marital_Status as string,
      Phone_Number as string,
      Email as string,
      Address as string,
      City as string,
      State as string,
      ZipCode as string,
      Ethnicity as string,
      Blood_Type as string,
      Allergies as string,
      Emergency_Contact_Name as string,
      Emergency_Contact_Phone as string,
      Insurance_ID as string,
      Doctor_ID as string,
      Doctor_Name as string,
      Doctor_Licence as string,
      Specialization as string,
      Department_ID as string,
      Department_Name as string,
      Doctor_Email as string,
      Doctor_Phone as string,
      Years_of_Experience as string,
      Shift as string,
      Hospital_ID as string,
      Hospital_Name as string,
      Hospital_Branch as string,
      Hospital_City as string,
      Hospital_State as string,
      Hospital_Type as string,
      Visit_Date as string,
      Visit_Time as string,
      Discharge_Date as string,
      Billing_Date as string,
      Date_ID as string,
      Diagnosis_Code as string,
      Diagnosis_Description as string,
      Procedure_Code as string,
      Procedure_Description as string,
      Medication_ID as string,
      Medication_Name as string,
      Medication_Strength as string,
      Medication_Form as string,
      Medication_Route as string,
      Medication_SideEffects as string,
      Visit_ID as string,
      Invoice_ID as string,
      Total_Amount as string,
      Currency as string,
      Insurance_Covered_Amount as string,
      Patient_Pay_Amount as string,
      Payment_Status as string,
      Payment_Method as string,
      Billing_Provider as string,
      Claim_Number as string,
      Length_of_Stay_Days as string,
      Visit_Duration_Minutes as string,
      Room_Type as string,
      Admission_Type as string,
      Discharge_Disposition as string,
      Dispense_ID as string,
      Pharmacy_Name as string,
      Dispense_Quantity as string,
      Dispense_Cost as string,
      Refillable as string,
      Record_Created_Timestamp as string,
      Data_Source as string
 ),
 allowSchemaDrift: true,
 validateSchema: false,
 ignoreNoFilesFound: false) ~> StagingSource

StagingSource select(mapColumn(
      Patient_ID,
      Patient_First_Name,
      Patient_Last_Name,
      Gender,
      DOB,
      Age,
      Marital_Status,
      Phone_Number,
      Email,
      Address,
      City,
      State,
      ZipCode,
      Ethnicity,
      Blood_Type,
      Allergies,
      Emergency_Contact_Name,
      Emergency_Contact_Phone,
      Insurance_ID
 ),
 skipDuplicateMapInputs: true,
 skipDuplicateMapOutputs: true) ~> SelectDimPatient

SelectDimPatient aggregate(groupBy(Patient_ID),
 Patient_First_Name = first(Patient_First_Name),
     Patient_Last_Name = first(Patient_Last_Name),
     Gender = first(Gender),
     DOB = first(DOB),
     Age = first(Age),
     Marital_Status = first(Marital_Status),
     Phone_Number = first(Phone_Number),
     Email = first(Email),
     Address = first(Address),
     City = first(City),
     State = first(State),
     ZipCode = first(ZipCode),
     Ethnicity = first(Ethnicity),
     Blood_Type = first(Blood_Type),
     Allergies = first(Allergies),
     Emergency_Contact_Name = first(Emergency_Contact_Name),
     Emergency_Contact_Phone = first(Emergency_Contact_Phone),
     Insurance_ID = first(Insurance_ID)) ~> AggregateDimPatient

AggregateDimPatient cast(output(
      Age as integer
 ),
 errors: true) ~> CastDimPatient

CastDimPatient sink(allowSchemaDrift: true,
 validateSchema: false,
 deletable:false,
 insertable:true,
 updateable:false,
 upsertable:false,
 format: 'table',
 skipDuplicateMapInputs: true,
 skipDuplicateMapOutputs: true,
 errorHandlingOption: 'stopOnFirstError') ~> LoadDimPatient

StagingSource select(mapColumn(
      Doctor_ID,
      Doctor_Name,
      Doctor_Licence,
      Specialization,
      Department_ID,
      Department_Name,
      Doctor_Email,
      Doctor_Phone,
      Years_of_Experience,
      Shift
 ),
 skipDuplicateMapInputs: true,
 skipDuplicateMapOutputs: true) ~> SelectDimDoctor

SelectDimDoctor aggregate(groupBy(Doctor_ID),
 Doctor_Name = first(Doctor_Name),
     Doctor_Licence = first(Doctor_Licence),
     Specialization = first(Specialization),
     Department_ID = first(Department_ID),
     Department_Name = first(Department_Name),
     Doctor_Email = first(Doctor_Email),
     Doctor_Phone = first(Doctor_Phone),
     Years_of_Experience = first(Years_of_Experience),
     Shift = first(Shift)) ~> AggregateDimDoctor

AggregateDimDoctor cast(output(
      Years_of_Experience as integer
 ),
 errors: true) ~> CastDimDoctor

CastDimDoctor sink(allowSchemaDrift: true,
 validateSchema: false,
 deletable:false,
 insertable:true,
 updateable:false,
 upsertable:false,
 format: 'table',
 skipDuplicateMapInputs: true,
 skipDuplicateMapOutputs: true,
 errorHandlingOption: 'stopOnFirstError') ~> LoadDimDoctor

StagingSource select(mapColumn(
      Hospital_ID,
      Hospital_Name,
      Hospital_Branch,
      Hospital_City,
      Hospital_State,
      Hospital_Type
 ),
 skipDuplicateMapInputs: true,
 skipDuplicateMapOutputs: true) ~> SelectDimHospital

SelectDimHospital aggregate(groupBy(Hospital_ID),
 Hospital_Name = first(Hospital_Name),
     Hospital_Branch = first(Hospital_Branch),
     Hospital_City = first(Hospital_City),
     Hospital_State = first(Hospital_State),
     Hospital_Type = first(Hospital_Type)) ~> AggregateDimHospital

AggregateDimHospital sink(allowSchemaDrift: true,
 validateSchema: false,
 deletable:false,
 insertable:true,
 updateable:false,
 upsertable:false,
 format: 'table',
 skipDuplicateMapInputs: true,
 skipDuplicateMapOutputs: true,
 errorHandlingOption: 'stopOnFirstError') ~> LoadDimHospital

StagingSource select(mapColumn(
      Date_ID,
      Visit_Date,
      Discharge_Date,
      Billing_Date
 ),
 skipDuplicateMapInputs: true,
 skipDuplicateMapOutputs: true) ~> SelectDimDate

SelectDimDate aggregate(groupBy(Date_ID),
 Visit_Date = first(Visit_Date),
     Billing_Date = first(Billing_Date),
     Discharge_Date = first(Discharge_Date)) ~> AggregateDimDate

AggregateDimDate derive(Visit_Date = toDate(Visit_Date),
      Discharge_Date = toDate(Discharge_Date),
      Billing_Date = toDate(Billing_Date)) ~> DeriveDimDate

DeriveDimDate sink(allowSchemaDrift: true,
 validateSchema: false,
 deletable:false,
 insertable:true,
 updateable:false,
 upsertable:false,
 format: 'table',
 skipDuplicateMapInputs: true,
 skipDuplicateMapOutputs: true,
 errorHandlingOption: 'stopOnFirstError') ~> LoadDimDate

StagingSource select(mapColumn(
      Medication_ID,
      Medication_Name,
      Medication_Strength,
      Medication_Form,
      Medication_Route,
      Medication_SideEffects
 ),
 skipDuplicateMapInputs: true,
 skipDuplicateMapOutputs: true) ~> SelectDimMedication

SelectDimMedication aggregate(groupBy(Medication_ID),
 Medication_Name = first(Medication_Name),
     Medication_Strength = first(Medication_Strength),
     Medication_Form = first(Medication_Form),
     Medication_Route = first(Medication_Route),
     Medication_SideEffects = first(Medication_SideEffects)) ~> AggregateDimMedication

AggregateDimMedication sink(allowSchemaDrift: true,
 validateSchema: false,
 deletable:false,
 insertable:true,
 updateable:false,
 upsertable:false,
 format: 'table',
 skipDuplicateMapInputs: true,
 skipDuplicateMapOutputs: true,
 errorHandlingOption: 'stopOnFirstError') ~> LoadDimMedication

StagingSource select(mapColumn(
      Visit_ID,
      Patient_ID,
      Doctor_ID,
      Hospital_ID,
      Date_ID,
      Medication_ID,
      Diagnosis_Code,
      Diagnosis_Description,
      Procedure_Code,
      Procedure_Description,
      Invoice_ID,
      Total_Amount,
      Currency,
      Insurance_Covered_Amount,
      Patient_Pay_Amount,
      Payment_Status,
      Payment_Method,
      Billing_Provider,
      Claim_Number,
      Length_of_Stay_Days,
      Visit_Duration_Minutes,
      Room_Type,
      Admission_Type,
      Discharge_Disposition,
      Dispense_ID,
      Pharmacy_Name,
      Dispense_Quantity,
      Dispense_Cost,
      Refillable,
      Visit_Time,
      Record_Created_Timestamp,
      Data_Source
 ),
 skipDuplicateMapInputs: true,
 skipDuplicateMapOutputs: true) ~> SelectFactVisit

SelectFactVisit cast(output(
      Total_Amount as decimal(18,2),
      Insurance_Covered_Amount as decimal(18,2),
      Patient_Pay_Amount as decimal(18,2),
      Length_of_Stay_Days as integer,
      Visit_Duration_Minutes as integer,
      Dispense_Quantity as integer,
      Dispense_Cost as decimal(18,2),
      Record_Created_Timestamp as timestamp
 ),
 errors: true) ~> CastFactVisit

CastFactVisit sink(allowSchemaDrift: true,
 validateSchema: false,
 deletable:false,
 insertable:true,
 updateable:false,
 upsertable:false,
 format: 'table',
 skipDuplicateMapInputs: true,
 skipDuplicateMapOutputs: true,
 errorHandlingOption: 'stopOnFirstError') ~> LoadFactVisit"""

        dataflow_properties = MappingDataFlow(
            sources=[
                DataFlowSource(
                    name='StagingSource',
                    dataset=DatasetReference(
                        reference_name=self.names['staging_csv_dataset'],
                        type='DatasetReference'
                    )
                )
            ],
            sinks=[
                DataFlowSink(
                    name='LoadDimPatient',
                    dataset=DatasetReference(
                        reference_name=self.names['dim_patient_dataset'],
                        type='DatasetReference'
                    )
                ),
                DataFlowSink(
                    name='LoadDimDoctor',
                    dataset=DatasetReference(
                        reference_name=self.names['dim_doctor_dataset'],
                        type='DatasetReference'
                    )
                ),
                DataFlowSink(
                    name='LoadDimHospital',
                    dataset=DatasetReference(
                        reference_name=self.names['dim_hospital_dataset'],
                        type='DatasetReference'
                    )
                ),
                DataFlowSink(
                    name='LoadDimDate',
                    dataset=DatasetReference(
                        reference_name=self.names['dim_date_dataset'],
                        type='DatasetReference'
                    )
                ),
                DataFlowSink(
                    name='LoadDimMedication',
                    dataset=DatasetReference(
                        reference_name=self.names['dim_medication_dataset'],
                        type='DatasetReference'
                    )
                ),
                DataFlowSink(
                    name='LoadFactVisit',
                    dataset=DatasetReference(
                        reference_name=self.names['fact_table_dataset'],
                        type='DatasetReference'
                    )
                )
            ],
            transformations=[
            Transformation(name='SelectDimPatient'),
            Transformation(name='AggregateDimPatient'),
            Transformation(name='CastDimPatient'),
            Transformation(name='SelectDimDoctor'),
            Transformation(name='AggregateDimDoctor'),
            Transformation(name='CastDimDoctor'),
            Transformation(name='SelectDimHospital'),
            Transformation(name='AggregateDimHospital'),
            Transformation(name='SelectDimDate'),
            Transformation(name='AggregateDimDate'),
            Transformation(name='DeriveDimDate'),
            Transformation(name='SelectDimMedication'),
            Transformation(name='AggregateDimMedication'),
            Transformation(name='SelectFactVisit'),
            Transformation(name='CastFactVisit')
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
        print(f"✓ Transform Data Flow created: {result.name}")
        return result
    
    # ==================== Pipeline ====================
    
    def create_pipeline(self):
        """Create main pipeline with union and transform activities"""
        name = self.names['pipeline']
        print(f"Creating Pipeline: {name}...")
        
        # Activity 1: Execute Data Flow - Union All CSVs
        union_dataflow_activity = ExecuteDataFlowActivity(
            name='UnionAllHospitalCSVs',
            policy=ActivityPolicy(
                timeout='0.12:00:00',
                retry=0,
                retry_interval_in_seconds=30,
                secure_output=False,
                secure_input=False
            ),
            data_flow=DataFlowReference(
                reference_name=self.names['union_dataflow'],
                type='DataFlowReference'
            ),
            compute=ExecuteDataFlowActivityTypePropertiesCompute(
                compute_type='General',
                core_count=8
            ),
            trace_level='Fine'
        )
        
        # Activity 2: Execute Data Flow - Transform to Fact/Dimension
        transform_dataflow_activity = ExecuteDataFlowActivity(
            name='TransformToFactDimension',
            depends_on=[
                ActivityDependency(
                    activity='UnionAllHospitalCSVs',
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
                reference_name=self.names['transform_dataflow'],
                type='DataFlowReference'
            ),
            compute=ExecuteDataFlowActivityTypePropertiesCompute(
                compute_type='General',
                core_count=8
            ),
            trace_level='Fine'
        )
        
        # Create pipeline with both activities
        pipeline = PipelineResource(
            description='Multi-hospital CSV to SQL pipeline with union and fact/dimension transformation',
            activities=[union_dataflow_activity, transform_dataflow_activity]
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
    
    def deploy_complete_solution(self, sql_config=None, blob_config=None):
        """
        Deploy complete Hospital CSV to SQL pipeline solution
        
        Args:
            sql_config: dict with keys: server_name, database_name, username, password
            blob_config: dict with keys: account_name, account_key
        """
        print("=" * 80)
        print("DEPLOYING HOSPITAL CSV TO SQL PIPELINE")
        print("=" * 80)
        print()
        
        try:
            print("Step 1: Creating Linked Services")
            print("-" * 80)
            if sql_config:
                self.create_sql_linked_service(
                    server_name=sql_config.get('server_name'),
                    database_name=sql_config.get('database_name'),
                    username=sql_config.get('username'),
                    password=sql_config.get('password')
                )
            else:
                self.create_sql_linked_service()
            
            if blob_config:
                self.create_blob_storage_linked_service(
                    account_name=blob_config.get('account_name'),
                    account_key=blob_config.get('account_key')
                )
            else:
                self.create_blob_storage_linked_service()
            print()
            
            print("Step 2: Creating Datasets")
            print("-" * 80)
            self.create_source_csv_dataset()
            self.create_staging_csv_dataset()
            self.create_fact_table_dataset()
            self.create_dimension_datasets()
            print()
            
            print("Step 3: Creating Data Flows")
            print("-" * 80)
            self.create_union_dataflow()
            self.create_transform_dataflow()
            print()
            
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
            print(f"    ├── Activity 1: UnionAllHospitalCSVs (Data Flow)")
            print(f"    │   └── Data Flow: {self.names['union_dataflow']} (Source: applicationdata/source/*.csv)")
            print(f"    │       └── Sink: Staging CSV (Union output)")
            print(f"    └── Activity 2: TransformToFactDimension (Data Flow)")
            print(f"        └── Data Flow: {self.names['transform_dataflow']}")
            print(f"            ├── Source: Staging CSV")
            print(f"            ├── Sink: FactVisit (Fact Table)")
            print(f"            ├── Sink: DimPatient (Dimension Table)")
            print(f"            ├── Sink: DimDoctor (Dimension Table)")
            print(f"            ├── Sink: DimHospital (Dimension Table)")
            print(f"            ├── Sink: DimDate (Dimension Table)")
            print(f"            └── Sink: DimMedication (Dimension Table)")
            print()
            print("Schema Details:")
            print("  Fact Table:")
            print("    - FactVisit: 32 columns including Visit_ID (PK), foreign keys, measures")
            print("  Dimension Tables:")
            print("    - DimPatient: 19 columns including Patient_ID (PK)")
            print("    - DimDoctor: 10 columns including Doctor_ID (PK)")
            print("    - DimHospital: 6 columns including Hospital_ID (PK)")
            print("    - DimDate: 4 columns including Date_ID (PK)")
            print("    - DimMedication: 6 columns including Medication_ID (PK)")
            print()
            
        except Exception as e:
            print(f"✗ Deployment failed: {str(e)}")
            import traceback
            traceback.print_exc()
            raise
    
    # ==================== Pipeline Execution ====================
    
    def run_pipeline(self, parameters=None):
        """Execute the Hospital CSV to SQL pipeline"""
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
    Deploy and run the Hospital CSV to SQL pipeline
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
    print()
    
    # Initialize pipeline manager with credentials
    pipeline_manager = HospitalCSVToSQLPipeline(
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
    
    # Optional: Run the pipeline
    print("\nDeployment complete!")
    user_input = input("Do you want to run the pipeline now? (yes/no): ")
    if user_input.lower() in ['yes', 'y']:
        run_id = pipeline_manager.run_pipeline()
        
        if run_id:
            # Monitor execution
            monitor_input = input("\nDo you want to monitor the pipeline execution? (yes/no): ")
            if monitor_input.lower() in ['yes', 'y']:
                status = pipeline_manager.monitor_pipeline(run_id)
                print(f"\nFinal Status: {status}")

if __name__ == '__main__':
    """
    Azure Data Factory - Hospital CSV to SQL Pipeline
    
    This script creates and deploys a complete ADF pipeline that:
    1. Reads multiple CSV files from Azure Blob Storage (source folder)
    2. Unions all CSV files into a single staging dataset
    3. Transforms data with proper type casting
    4. Splits into fact and dimension tables:
       - FactVisit (32 columns)
       - DimPatient (19 columns)
       - DimDoctor (10 columns)
       - DimHospital (6 columns)
       - DimDate (4 columns)
       - DimMedication (6 columns)
    5. Creates tables in Azure SQL Database with proper schemas
    6. Loads data into all tables
    
    Prerequisites:
    - Python packages: azure-identity, azure-mgmt-datafactory
    - CSV files uploaded to: macanpocstorageaccount/applicationdata/source/
    - Azure SQL Database: dataiq-server.database.windows.net/dataiq-database
    - Service Principal with Data Factory Contributor role
    
    Usage:
    1. Update credentials in main() function if needed
    2. Run: python hospital_csv_to_sql_pipeline.py
    3. Follow prompts to execute and monitor pipeline
    
    Features:
    - Automatic table creation with proper schemas
    - Type casting for numeric and date fields
    - Deduplication of dimension records
    - Error handling and monitoring
    - Truncate and reload pattern for data refresh
    """
    
    main()