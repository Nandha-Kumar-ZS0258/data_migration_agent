# app.py
import streamlit as st
import pandas as pd
import json
from azure_services.azure_helpers import AzureServices
from agents.openai_agents import AzureOpenAIAgents
import time

# ==================== PAGE CONFIGURATION ====================

st.set_page_config(
    page_title="ADF Pipeline Generator",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🤖 Agentic AI - Data Migration ")
st.markdown("---")

# ==================== HELPER FUNCTIONS ====================

def get_cached_table_schemas(selected_tables, selected_schemas, azure_services):
    """
    Helper function to batch fetch table schemas with caching.
    Returns a dictionary mapping table names to their schema information.
    """
    if not selected_tables or not selected_schemas:
        return {}
    
    # Create cache key based on selected tables
    cache_key = f"target_tables_{'_'.join(sorted(selected_tables))}"
    
    # Check if already cached
    if cache_key in st.session_state:
        return st.session_state[cache_key]
    
    # Ensure cached_tables exists
    if 'cached_tables' not in st.session_state:
        st.session_state.cached_tables = {}
    
    # Build schema->tables map using cached data
    schema_tables_map = {}
    for schema in selected_schemas:
        if schema not in st.session_state.cached_tables:
            st.session_state.cached_tables[schema] = azure_services.get_tables_by_schema(schema)
        schema_tables_map[schema] = st.session_state.cached_tables[schema]
    
    # Fetch table schemas
    target_tables = {}
    for table in selected_tables:
        # Find which schema this table belongs to
        table_schema = None
        for schema, tables_list in schema_tables_map.items():
            if table in tables_list:
                table_schema = schema
                break
        
        if table_schema:
            # Cache individual table schemas
            schema_key = f"schema_{table_schema}_{table}"
            if schema_key not in st.session_state:
                st.session_state[schema_key] = azure_services.get_table_schema(table_schema, table)
            target_tables[table] = st.session_state[schema_key]
    
    # Cache the result
    st.session_state[cache_key] = target_tables
    return target_tables

# Initialize session state
if 'step' not in st.session_state:
    st.session_state.step = 1

# Initialize Azure Services with error handling
if 'azure_services' not in st.session_state:
    try:
        st.session_state.azure_services = AzureServices()
    except ValueError as e:
        st.error(f"❌ Azure Configuration Error: {str(e)}")
        st.markdown("---")
        st.markdown("### 🔧 Configuration Instructions")
        st.markdown("""
        **For Azure Web App Deployment:**
        
        1. Go to Azure Portal → Your Web App → **Configuration** → **Application Settings**
        2. Add the following environment variables:
        
        **Required Azure Credentials:**
        - `AZURE_TENANT_ID` - Your Azure AD Tenant ID
        - `AZURE_CLIENT_ID` - Your Service Principal Client ID  
        - `AZURE_CLIENT_SECRET` - Your Service Principal Secret
        - `AZURE_SUBSCRIPTION_ID` - Your Azure Subscription ID
        
        **Additional Required Variables:**
        - `AZURE_RESOURCE_GROUP` - Your resource group name
        - `AZURE_DATA_FACTORY` - Your Data Factory name
        - `AZURE_LOCATION` - Azure region (e.g., `eastus`)
        - `AZURE_SQL_SERVER` - SQL server (e.g., `server.database.windows.net`)
        - `AZURE_SQL_DATABASE` - Database name
        - `AZURE_SQL_USER` - SQL username
        - `AZURE_SQL_PASSWORD` - SQL password
        - `AZURE_STORAGE_ACCOUNT` - Storage account name
        - `AZURE_STORAGE_KEY` - Storage account key
        - `AZURE_OPENAI_KEY` - Azure OpenAI API key
        - `AZURE_OPENAI_ENDPOINT` - Azure OpenAI endpoint URL
        - `AZURE_OPENAI_API_VERSION` - API version (e.g., `2024-02-15-preview`)
        - `AZURE_OPENAI_DEPLOYMENT` - Deployment name/model (e.g., `gpt-4o-mini`)
        
        3. Click **Save** and **Restart** the Web App
        
        **For Local Development:**
        - Create `.streamlit/secrets.toml` file with the same variables
        - Or set them as environment variables
        """)
        st.stop()

if 'openai_agents' not in st.session_state:
    try:
        st.session_state.openai_agents = AzureOpenAIAgents()
        # Check if initialization actually succeeded
        if hasattr(st.session_state.openai_agents, 'client') and st.session_state.openai_agents.client is None:
            if hasattr(st.session_state.openai_agents, 'init_error'):
                st.warning(f"⚠️ OpenAI Agent initialized but client is None: {st.session_state.openai_agents.init_error}")
            else:
                st.warning("⚠️ OpenAI Agent initialized but client is None. Some features may not work.")
    except Exception as e:
        st.warning(f"⚠️ OpenAI Agent initialization warning: {str(e)}")
        # Always set the attribute, even if initialization failed
        # Create a minimal object that won't crash when accessed
        try:
            # Try to create the object anyway - it might have partial initialization
            st.session_state.openai_agents = AzureOpenAIAgents()
        except:
            # If even creating the object fails, create a fallback object
            class FallbackOpenAIAgent:
                def __init__(self, error_msg):
                    self.client = None
                    self.model = None
                    self.init_error = f"OpenAI client failed to initialize: {error_msg}"
                    self._sample_code_reference_cache = None
                
                def detect_column_datatypes(self, csv_data, agent1_analysis=None, target_tables=None, stream_container=None):
                    return self._create_fallback_datatypes(csv_data, agent1_analysis)
                
                def _create_fallback_datatypes(self, csv_data, agent1_analysis=None):
                    """Fallback implementation using pandas dtypes"""
                    import pandas as pd
                    result = {
                        'reasoning': 'Fallback: Using heuristic data type detection (OpenAI unavailable)',
                        'columns': {}
                    }
                    for col in csv_data.columns:
                        dtype = str(csv_data[col].dtype)
                        if 'int' in dtype:
                            sql_type = 'integer'
                        elif 'float' in dtype:
                            sql_type = 'decimal(10,2)'
                        elif 'datetime' in dtype or 'date' in dtype:
                            sql_type = 'date'
                        else:
                            sql_type = 'varchar(255)'
                        result['columns'][col] = {
                            'sql_type': sql_type,
                            'reasoning': f'Inferred from pandas dtype: {dtype}'
                        }
                    return result
                
                def analyze_csv_structure_v2(self, csv_data, csv_filename=None, target_tables=None, stream_container=None):
                    return {
                        'reasoning': 'Fallback: OpenAI unavailable - using basic structure analysis',
                        'tables': [],
                        'fact_tables': [],
                        'dimension_tables': []
                    }
                
                def generate_python_sdk_code(self, *args, **kwargs):
                    return {
                        'code': '# Code generation unavailable - OpenAI client not initialized',
                        'validation_result': {'is_valid': False, 'issues': ['OpenAI client not available']}
                    }
            
            st.session_state.openai_agents = FallbackOpenAIAgent(str(e))

# ==================== MAIN CONTENT AREA ====================

# Create tabs
tab1, tab2, tab3, tab5 = st.tabs(
    ["📊 Source & Target Connection", "🔄 Agent Processing", "💾 Generated Code", "📥 Download"]
)

# ==================== TAB 1: SOURCE ANALYSIS ====================

with tab1:
    st.header("📊 Source CSV Analysis")
    
    # ==================== CONFIGURATION SECTION ====================
    st.subheader("⚙️ Configuration")
    
    # Step 1: Source Configuration
    st.markdown("#### Step 1️⃣: Source Configuration")
    
    # Upload Source files button
    uploaded_file = st.file_uploader(
        "Upload Source files",
        type=['csv'],
        help="Upload a CSV file to blob storage"
    )
    
    if uploaded_file is not None:
        if st.button("📤 Upload to Blob Storage", key="upload_btn"):
            container_name = "applicationdata"
            folder_name = "source"
            file_name = uploaded_file.name
            
            # Show uploading message
            upload_status = st.info("⏳ Uploading file to blob storage...")
            
            # Read file data
            file_data = uploaded_file.read()
            
            # Upload to blob
            success, message = st.session_state.azure_services.upload_csv_to_blob(
                container_name,
                folder_name,
                file_data,
                file_name
            )
            
            # Clear the uploading message
            upload_status.empty()
            
            if success:
                st.success(f"✅ {message}")
                # Refresh CSV files list
                if 'csv_files' in st.session_state:
                    del st.session_state.csv_files
            else:
                st.error(f"❌ {message}")
    
    col1, col2 = st.columns(2)
    
    with col1:
        source_container = st.text_input(
            "Source Container",
            value=st.session_state.get('source_container', 'applicationdata'),
            help="Blob Storage container name",
            key="source_container_input"
        )
        st.session_state.source_container = source_container
    
    with col2:
        source_folder = st.text_input(
            "Source Folder",
            value=st.session_state.get('source_folder', 'source'),
            help="Folder path inside container",
            key="source_folder_input"
        )
        st.session_state.source_folder = source_folder
    
    col_refresh, col_list = st.columns([1, 4])
    
    with col_refresh:
        if st.button("🔄 Refresh Cache", key="refresh_cache_btn", help="Clear all cached data and reload"):
            # Clear Streamlit caches
            st.cache_data.clear()
            st.cache_resource.clear()
            
            # Clear session state caches
            keys_to_remove = [k for k in st.session_state.keys() 
                             if k.startswith('cached_') or k.startswith('target_tables_') 
                             or k.startswith('schema_')]
            for key in keys_to_remove:
                del st.session_state[key]
            
            st.success("✅ Cache cleared! Reloading...")
            st.rerun()
    
    with col_list:
        if st.button("🔍 List CSV Files", key="list_csv_btn"):
            csv_files = st.session_state.azure_services.list_csv_files_in_blob(
                source_container,
                source_folder
            )
            st.session_state.csv_files = csv_files
            st.success(f"Found {len(csv_files)} CSV files")
    
    if 'csv_files' in st.session_state and st.session_state.csv_files:
        selected_csv = st.selectbox(
            "Select CSV File",
            st.session_state.csv_files,
            help="Choose CSV file to analyze",
            key="csv_selectbox"
        )
        st.session_state.selected_csv = selected_csv
    
    st.markdown("---")
    
    # ==================== LOAD AND PREVIEW CSV SECTION ====================
    st.subheader("📖 Load and Preview CSV")
    
    if 'selected_csv' in st.session_state and st.session_state.selected_csv:
        if st.button("📖 Load and Preview CSV", key="load_csv_btn", type="primary"):
            with st.spinner("Reading CSV from Blob Storage..."):
                df = st.session_state.azure_services.read_csv_from_blob(
                    st.session_state.source_container,
                    st.session_state.selected_csv
                )
                
                if df is not None:
                    st.session_state.csv_data = df
                    st.success("CSV loaded successfully!")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Total Rows", df.shape[0])
                    col2.metric("Total Columns", df.shape[1])
                    col3.metric("Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024:.2f} KB")
                    col4.metric("Null Values", df.isnull().sum().sum())
                    
                    st.subheader("Column Information")
                    col_info = pd.DataFrame({
                        "Column": df.columns,
                        "Type": df.dtypes.astype(str),
                        "Non-Null": df.count(),
                        "Null": df.isnull().sum(),
                        "Unique": df.nunique()
                    })
                    st.dataframe(col_info, use_container_width=True)
                    
                    st.subheader("Data Preview")
                    st.dataframe(df.head(10), use_container_width=True)
                else:
                    st.error("Failed to load CSV")
    else:
        st.info("👈 Please configure source files above and select a CSV file first")
    
    st.markdown("---")
    
    # Step 2: Destination Configuration
    st.markdown("#### Step 2️⃣: Destination Configuration")
    
    # Test database connectivity first
    db_accessible = False
    try:
        engine = st.session_state.azure_services.get_sql_engine()
        if engine is not None:
            with engine.connect() as conn:
                conn.execute("SELECT 1")
            db_accessible = True
    except:
        db_accessible = False
    
    # Network test button
    if st.button("🌐 Test Network Connectivity", key="test_network_btn"):
        with st.spinner("Testing network connectivity..."):
            success, message = st.session_state.azure_services.test_network_connectivity()
            if success:
                st.success(f"✅ {message}")
            else:
                st.error(f"❌ {message}")
    
    # Multi-select schemas - Cache in session state
    if 'cached_schemas' not in st.session_state:
        with st.spinner("Loading schemas..."):
            st.session_state.cached_schemas = st.session_state.azure_services.get_all_schemas()
    
    schemas = st.session_state.cached_schemas
    # Set 'dbo' as default schema if it exists, otherwise use first schema
    default_schema = 'dbo' if 'dbo' in schemas else (schemas[0] if schemas else [])
    selected_schemas = st.multiselect(
        "Select Schemas",
        schemas,
        default=[default_schema] if default_schema and isinstance(default_schema, str) else [],
        help="Select one or more database schemas",
        key="schemas_multiselect"
    )
    
    # Get tables for selected schemas - Cache per schema in session state
    if 'cached_tables' not in st.session_state:
        st.session_state.cached_tables = {}
    
    all_selected_tables = []
    if selected_schemas:
        st.write("📊 **Available Tables:**")
        
        for schema in selected_schemas:
            # Cache tables per schema
            if schema not in st.session_state.cached_tables:
                st.session_state.cached_tables[schema] = st.session_state.azure_services.get_tables_by_schema(schema)
            
            tables = st.session_state.cached_tables[schema]
            if tables:
                st.write(f"**{schema}:**")
                selected_tables_for_schema = st.multiselect(
                    f"Tables in {schema}",
                    tables,
                    default=[],  # No tables selected by default - user must select manually
                    key=f"tables_{schema}",
                    help=f"Select tables from {schema} schema"
                )
                all_selected_tables.extend(selected_tables_for_schema)
            else:
                if db_accessible:
                    st.write(f"**{schema}:** No tables found")
                else:
                    st.write(f"**{schema}:** 🔧 Offline mode - showing expected tables")
                    st.write(", ".join(tables) if tables else "No tables found")
    else:
        st.info("👈 Please select at least one schema")
    
    # Store selected schemas and tables in session state
    st.session_state.selected_schemas = selected_schemas
    st.session_state.selected_tables = all_selected_tables
    
    # Display summary
    if selected_schemas and all_selected_tables:
        st.success(f"✅ Selected {len(selected_schemas)} schema(s) with {len(all_selected_tables)} table(s)")
        st.write("**Selected Tables:**")
        for table in all_selected_tables:
            st.write(f"• {table}")
    elif selected_schemas:
        st.warning("⚠️ No tables selected")

# ==================== TAB 2: AGENT PROCESSING ====================

with tab2:
    st.header("🔄 Agent Processing Pipeline")
    
    # Show selected tables summary
    if st.session_state.selected_tables:
        st.info(f"🎯 **Target Tables:** {', '.join(st.session_state.selected_tables)}")
    else:
        st.warning("⚠️ No tables selected. Please select tables from the sidebar.")
    
    if 'csv_data' in st.session_state:
        
        col1, col2, col3 = st.columns(3)
        
        # Agent 1: Data Type Detection (formerly Agent 2)
        with col1:
            st.subheader("Agent 1️⃣: Data Type Detection")
            st.write("Identifies optimal SQL data types for each column")
            
            if st.button("🚀 Run Agent 1", key="agent1_btn"):
                # Display initial message
                st.info("Agent 1 detecting data types....")
                st.text("🔄 Starting Agent 1: Data Type Detection...")
                
                # Create streaming container for Agent 1
                with st.expander("🔴 Live AI Response (Agent 1)", expanded=False):
                    st.markdown("### 🤖 Streaming Response:")
                    agent1_stream_area = st.empty()
                
                # Ultimate safety wrapper - catch ANY exception that might escape
                try:
                    with st.spinner("Agent 1 detecting data types..."):
                        st.text("📊 Analyzing CSV data structure...")
                        time.sleep(0.5)  # Small delay for log visibility
                        
                        # Get target tables if selected - Use cached helper function
                        target_tables = {}
                        if st.session_state.selected_tables:
                            st.text(f"🔍 Found {len(st.session_state.selected_tables)} target table(s)...")
                            target_tables = get_cached_table_schemas(
                                st.session_state.selected_tables,
                                st.session_state.selected_schemas,
                                st.session_state.azure_services
                            )
                            st.text("✅ Target tables schema retrieved")
                        
                        st.text("🤖 Calling OpenAI API for data type detection...")
                        # Safety check
                        if not hasattr(st.session_state, 'openai_agents') or st.session_state.openai_agents is None:
                            raise AttributeError("OpenAI agents not initialized")
                        result = st.session_state.openai_agents.detect_column_datatypes(
                            st.session_state.csv_data,
                            agent1_analysis=None,  # No dependency on CSV analysis anymore
                            target_tables=target_tables if target_tables else None,
                            stream_container=agent1_stream_area
                        )
                        
                        st.text("✅ Data type detection completed")
                        
                        # Result is always returned (never None, never raises)
                        # Store as agent1_result (but it's actually data type detection)
                        st.session_state.agent1_result = result
                        st.session_state.agent1_datatype_result = result  # Also store with explicit name
                        
                        # Check if it's fallback analysis
                        if result.get('columns') and any('fallback' in str(v.get('reasoning', '')).lower() for v in result.get('columns', {}).values()):
                            st.text("⚠️ Using fallback analysis (AI unavailable)")
                            st.warning("⚠️ Agent 1 Complete (using fallback analysis)")
                            st.info("💡 AI analysis unavailable - using heuristic data type detection based on pandas dtypes")
                        else:
                            st.text("✅ AI analysis successful")
                            st.success("✅ Agent 1 Complete")
                        
                        st.text("📋 Displaying results...")
                        
                except Exception as e:
                    st.text(f"❌ Error occurred: {type(e).__name__}")
                    # Ultimate safety net - if function somehow raises, use fallback directly
                    st.error(f"❌ Unexpected error (should not happen): {type(e).__name__}: {e}")
                    st.info("💡 Using fallback analysis...")
                    try:
                        st.text("📊 Attempting fallback analysis...")
                        # Safety check
                        if not hasattr(st.session_state, 'openai_agents') or st.session_state.openai_agents is None:
                            raise AttributeError("OpenAI agents not initialized")
                        fallback_result = st.session_state.openai_agents._create_fallback_datatypes(
                            st.session_state.csv_data,
                            agent1_analysis=None
                        )
                        st.session_state.agent1_result = fallback_result
                        st.session_state.agent1_datatype_result = fallback_result
                        st.text("✅ Fallback analysis completed")
                        st.warning("⚠️ Agent 1 Complete (using fallback analysis)")
                    except Exception as fallback_err:
                        st.text(f"❌ Fallback also failed: {str(fallback_err)}")
                        st.error(f"❌ Even fallback failed: {fallback_err}")
                        st.stop()
                
                # Display result
                if 'result' in locals():
                    st.json(result)
        
        # Agent 2: CSV Structure Analysis (formerly Agent 1)
        with col2:
            st.subheader("Agent 2️⃣: CSV Data Modeler")
            st.write("Analyzes structure and suggests Fact/Dimension split")
            
            if st.button("🚀 Run Agent 2", key="agent2_btn"):
                # Display initial message
                st.info("Agent 2 analyzing CSV structure....")
                st.text("🔄 Starting Agent 2: CSV Analysis...")
                
                # Create streaming container for Agent 2
                with st.expander("🔴 Live AI Response (Agent 2)", expanded=False):
                    st.markdown("### 🤖 Streaming Response:")
                    agent2_stream_area = st.empty()
                
                try:
                    with st.spinner("Agent 2 analyzing CSV structure..."):
                        st.text("📊 Loading CSV data...")
                        time.sleep(0.5)
                        
                        # Get target tables if selected - Use cached helper function
                        target_tables = {}
                        if st.session_state.selected_tables:
                            st.text(f"🔍 Found {len(st.session_state.selected_tables)} target table(s)...")
                            target_tables = get_cached_table_schemas(
                                st.session_state.selected_tables,
                                st.session_state.selected_schemas,
                                st.session_state.azure_services
                            )
                            st.text("✅ Target tables schema retrieved")
                        
                        st.text("🤖 Calling OpenAI API for CSV structure analysis...")
                        # Safety check
                        if not hasattr(st.session_state, 'openai_agents') or st.session_state.openai_agents is None:
                            raise AttributeError("OpenAI agents not initialized")
                        result = st.session_state.openai_agents.analyze_csv_structure_v2(
                            st.session_state.csv_data,
                            st.session_state.selected_csv,
                            target_tables=target_tables if target_tables else None,
                            stream_container=agent2_stream_area
                        )
                        
                        st.text("✅ CSV structure analysis completed")
                        
                        # Result is always returned (never None)
                        # Store as agent2_result (but it's actually CSV analysis)
                        st.session_state.agent2_result = result
                        st.session_state.agent2_csv_result = result  # Also store with explicit name
                        
                        # Check if it's fallback analysis
                        reasoning = result.get('reasoning', '')
                        is_fallback = False
                        if isinstance(reasoning, str) and 'fallback' in reasoning.lower():
                            is_fallback = True
                        
                        if is_fallback:
                            st.text("⚠️ Using fallback analysis (AI unavailable)")
                            st.warning("⚠️ Agent 2 Complete (using fallback analysis)")
                            st.info("💡 AI analysis unavailable - using heuristic analysis based on column patterns")
                        else:
                            st.text("✅ AI analysis successful")
                            st.success("✅ Agent 2 Complete")
                        
                        st.text("📋 Displaying results...")
                        
                except ValueError as e:
                    st.text(f"❌ Configuration Error: {str(e)}")
                    st.error(f"❌ Agent 2 Configuration Error: {str(e)}")
                    st.info("💡 Please check your OpenAI configuration in `.streamlit/secrets.toml`")
                except Exception as e:
                    st.text(f"❌ Error: {str(e)}")
                    st.error(f"❌ Agent 2 Failed: {str(e)}")
                    st.info("💡 Check the error message above and verify your OpenAI API key and endpoint are correct")
                
                # Display result
                if 'result' in locals():
                    st.json(result)
        
        # Agent 3: Code Generation
        with col3:
            st.subheader("Agent 3️⃣: Code Generation")
            st.write("Generates complete Python SDK code")
            
            # Check if both agents have been run (note: agent1_result is now datatypes, agent2_result is CSV analysis)
            csv_result = st.session_state.get('agent2_result') or st.session_state.get('agent2_csv_result')
            datatype_result = st.session_state.get('agent1_result') or st.session_state.get('agent1_datatype_result')
            
            if csv_result and datatype_result:
                if st.session_state.selected_tables:
                    if st.button("🚀 Run Agent 3", key="agent3_btn"):
                        # Display initial message
                        st.info("Agent 3 generating Python SDK code....")
                        st.text("🔄 Starting Agent 3: Code Generation...")
                        
                        with st.spinner("Agent 3 generating Python SDK code..."):
                            st.text("📊 Gathering agent results...")
                            time.sleep(0.5)
                            
                            # Get destination table schemas for all selected tables - Use cached helper function
                            dest_tables = {}
                            if st.session_state.selected_tables:
                                st.text(f"🔍 Processing {len(st.session_state.selected_tables)} destination table(s)...")
                                target_tables_dict = get_cached_table_schemas(
                                    st.session_state.selected_tables,
                                    st.session_state.selected_schemas,
                                    st.session_state.azure_services
                                )
                                # Convert to dest_tables format (schema.table format)
                                for table, schema_info in target_tables_dict.items():
                                    # Find schema for this table
                                    table_schema = None
                                    if 'cached_tables' in st.session_state:
                                        for schema in st.session_state.selected_schemas:
                                            if schema in st.session_state.cached_tables and table in st.session_state.cached_tables[schema]:
                                                table_schema = schema
                                                break
                                    if table_schema:
                                        dest_tables[f"{table_schema}.{table}"] = schema_info
                                st.text("✅ Destination table schemas retrieved")
                            else:
                                st.warning("⚠️ No tables selected for code generation")
                                dest_tables = {}
                        
                            st.text("⚙️ Preparing Azure configuration...")
                            # Prepare Azure config
                            azure_config = {
                                'subscription_id': st.secrets.get('AZURE_SUBSCRIPTION_ID', 'XXXXX'),
                                'resource_group': st.secrets.get('AZURE_RESOURCE_GROUP', 'XXXXX'),
                                'factory_name': st.secrets.get('AZURE_DATA_FACTORY', 'XXXXX'),
                                'storage_account': st.secrets.get('AZURE_STORAGE_ACCOUNT', 'XXXXX'),
                                'storage_key': st.secrets.get('AZURE_STORAGE_KEY', 'XXXXX'),
                                'sql_server': st.secrets.get('AZURE_SQL_SERVER', 'XXXXX'),
                                'sql_database': st.secrets.get('AZURE_SQL_DATABASE', 'XXXXX'),
                                'sql_user': st.secrets.get('AZURE_SQL_USER', 'XXXXX'),
                                'sql_password': st.secrets.get('AZURE_SQL_PASSWORD', 'XXXXX'),
                                'tenant_id': st.secrets.get('AZURE_TENANT_ID', 'XXXXX'),
                                'client_id': st.secrets.get('AZURE_CLIENT_ID', 'XXXXX'),
                                'client_secret': st.secrets.get('AZURE_CLIENT_SECRET', 'XXXXX'),
                                'location': st.secrets.get('AZURE_LOCATION', 'East US')
                            }
                            st.text("✅ Azure configuration prepared")
                            
                            st.text("🤖 Calling OpenAI API for code generation...")
                            
                            # Initialize validation status
                            if 'validation_status' not in st.session_state:
                                st.session_state.validation_status = {
                                    'attempt': 0,
                                    'max_attempts': 3,
                                    'is_validating': False,
                                    'validation_passed': False,
                                    'issues': []
                                }
                            
                            # Create status container for validation feedback
                            status_container = st.container()
                            
                            # Create streaming container for Agent 3 code generation
                            with st.expander("🔴 Live Code Generation (Agent 3B)", expanded=True):
                                st.markdown("### 🤖 Streaming Code Generation:")
                                agent3_stream_area = st.empty()
                            
                            # Generate code
                            try:
                                # Note: csv_result is agent2_result, datatype_result is agent1_result
                                with status_container:
                                    st.info("🔄 Agent 3A: Generating decision logic...")
                                
                                # Safety check
                                if not hasattr(st.session_state, 'openai_agents') or st.session_state.openai_agents is None:
                                    raise AttributeError("OpenAI agents not initialized")
                                result = st.session_state.openai_agents.generate_python_sdk_code(
                                    csv_result,  # CSV analysis (was agent1_result)
                                    datatype_result,  # Data types (was agent2_result)
                                    dest_tables,
                                    azure_config,
                                    csv_data=st.session_state.csv_data,
                                    blob_container=st.session_state.get('source_container', 'applicationdata'),
                                    blob_folder=st.session_state.get('source_folder', 'source'),
                                    csv_filename=st.session_state.get('selected_csv'),  # CSV file path from frontend UI
                                    stream_container=agent3_stream_area
                                )
                                
                                # Handle both new format (dict) and old format (string) for backwards compatibility
                                if isinstance(result, dict):
                                    generated_code = result.get('code', '')
                                    validation_result = result.get('validation_result', {})
                                    attempt_count = result.get('attempt_count', 0)
                                    final_validation_status = result.get('final_validation_status', False)
                                else:
                                    # Old format - just code string
                                    generated_code = result
                                    validation_result = {
                                        "is_valid": True,
                                        "issues": [],
                                        "feedback": "Validation information not available (old format)",
                                        "validation_details": {}
                                    }
                                    attempt_count = 1
                                    final_validation_status = True
                                
                                if generated_code:
                                    st.session_state.generated_code = generated_code
                                    st.session_state.validation_result = validation_result
                                    st.session_state.validation_attempt_count = attempt_count
                                    st.session_state.final_validation_status = final_validation_status
                                    
                                    # Display success message
                                    with status_container:
                                        st.success("✅ Agent 3 Complete - Code Generated!")
                                    
                                    st.markdown("---")
                                    st.text("✅ Code generation completed successfully")
                                    st.text(f"📝 Generated {len(generated_code.split(chr(10)))} lines of code")
                                    st.success("🎉 **Code is ready!** You can proceed to Tab 3 to view the code and deploy.")
                                else:
                                    st.text("❌ Code generation failed - no code returned")
                                    st.error("❌ Agent 3 Failed - No code generated. Check console for details.")
                            except Exception as e:
                                st.text(f"❌ Error: {type(e).__name__}: {str(e)}")
                                st.error(f"❌ Agent 3 Failed: {type(e).__name__}: {str(e)}")
                                import traceback
                                st.code(traceback.format_exc())
                                st.session_state.generated_code = None
                            
                            st.text("📋 Displaying results...")
                else:
                    st.warning("⚠️ Please select tables from the sidebar first")
            else:
                st.warning("⚠️ Run Agent 1 and Agent 2 first")
    else:
        st.info("👈 Load CSV file from Tab 1 first")

# ==================== TAB 3: GENERATED CODE ====================

with tab3:
    st.header("💾 Generated Python SDK Code")
    
    if 'generated_code' in st.session_state:
        # Code Syntax Validation (Additional check)
        st.subheader("🔍 Code Syntax Validation")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("✅ Validate Code Syntax", key="syntax_check_btn"):
                try:
                    compile(st.session_state.generated_code, '<string>', 'exec')
                    st.success("✅ Code syntax is valid!")
                except SyntaxError as e:
                    st.error(f"❌ Syntax Error at line {e.lineno}: {e.msg}")
                    st.code(st.session_state.generated_code.split('\n')[e.lineno - 1], language="python")
        
        with col2:
            st.info("💡 **Tip:** This is a syntax validation check. Review the code before deployment.")
        
        st.markdown("---")
        
        # Display code
        st.subheader("📝 Generated Code")
        st.code(st.session_state.generated_code, language="python")
        
        # Deployment and Pipeline Execution Section
        st.markdown("---")
        st.subheader("🚀 Deployment & Pipeline Execution")
        
        deploy_col, pipeline_col = st.columns(2)
        
        with deploy_col:
            if st.button("🚀 Execute Code", key="deploy_btn", type="primary", use_container_width=True):
                with st.spinner("Deploying resources to Azure Data Factory..."):
                    success, message = st.session_state.azure_services.deploy_generated_code(
                        st.session_state.generated_code
                    )
                    
                    if success:
                        st.session_state.deployment_success = True
                        st.success(f"✅ {message}")
                        # Clear auto-fix flag if it was set
                        if 'code_auto_fixed' in st.session_state:
                            del st.session_state.code_auto_fixed
                    else:
                        st.session_state.deployment_success = False
                        st.error(f"❌ {message}")
                        # If code was auto-fixed, suggest retry
                        if st.session_state.get('code_auto_fixed', False):
                            st.info("💡 Code was auto-fixed. Please click 'Execute Code' again to deploy.")
                            st.session_state.code_auto_fixed = False
        
        with pipeline_col:
            # Check if deployment was successful
            deployment_ready = st.session_state.get('deployment_success', False)
            
            if deployment_ready:
                if st.button("▶️ Start Pipeline", key="start_pipeline_btn", use_container_width=True):
                    # Get Azure config from secrets
                    try:
                        resource_group = st.secrets.get('AZURE_RESOURCE_GROUP', '')
                        factory_name = st.secrets.get('AZURE_DATA_FACTORY', '')
                        
                        # Try to extract pipeline name from stored names or code
                        pipeline_name = None
                        
                        # First, try from stored pipeline names
                        if 'pipeline_names' in st.session_state:
                            pipeline_names = st.session_state.pipeline_names
                            if isinstance(pipeline_names, dict) and 'pipeline' in pipeline_names:
                                pipeline_name = pipeline_names['pipeline']
                        
                        # If not found, extract from code
                        if not pipeline_name:
                            pipeline_name = st.session_state.azure_services.get_pipeline_name_from_code(
                                st.session_state.generated_code
                            )
                        
                        # Final fallback
                        if not pipeline_name:
                            pipeline_name = "Pipeline"  # Default fallback
                        
                        with st.spinner("Starting pipeline execution..."):
                            run_id, message = st.session_state.azure_services.run_pipeline(
                                resource_group,
                                factory_name,
                                pipeline_name
                            )
                            
                            if run_id:
                                st.session_state.pipeline_run_id = run_id
                                st.session_state.pipeline_started = True
                                st.success(f"✅ Pipeline started! Run ID: {run_id}")
                            else:
                                st.error(f"❌ {message}")
                    except Exception as e:
                        st.error(f"❌ Error starting pipeline: {str(e)}")
            else:
                st.button("▶️ Start Pipeline", key="start_pipeline_btn_disabled", disabled=True, use_container_width=True,
                         help="Deploy code first before starting pipeline")
        
        # Pipeline Status Display
        if 'pipeline_started' in st.session_state and st.session_state.pipeline_started:
            st.markdown("---")
            st.subheader("📊 Pipeline Status")
            
            run_id = st.session_state.get('pipeline_run_id')
            if run_id:
                # Get current status
                try:
                    resource_group = st.secrets.get('AZURE_RESOURCE_GROUP', '')
                    factory_name = st.secrets.get('AZURE_DATA_FACTORY', '')
                    
                    status, status_message = st.session_state.azure_services.get_pipeline_status(
                        resource_group,
                        factory_name,
                        run_id
                    )
                    
                    if status:
                        # Display status with appropriate styling
                        status_col1, status_col2 = st.columns([1, 3])
                        
                        with status_col1:
                            if status == 'Succeeded':
                                st.success(f"✅ Status: {status}")
                            elif status == 'Failed':
                                st.error(f"❌ Status: {status}")
                            elif status == 'Cancelled':
                                st.warning(f"⚠️ Status: {status}")
                            else:
                                st.info(f"🔄 Status: {status}")
                        
                        with status_col2:
                            if status_message:
                                st.write(status_message)
                            st.write(f"**Run ID:** {run_id}")
                        
                        # Manual refresh button for status
                        if status in ['Queued', 'InProgress', 'Running']:
                            if st.button("🔄 Refresh Status", key="refresh_status_btn"):
                                st.rerun()
                    else:
                        st.warning(f"⚠️ {status_message}")
                except Exception as e:
                    st.error(f"❌ Error getting pipeline status: {str(e)}")
    else:
        st.info("👈 Generate code from Tab 2 first")

# ==================== TAB 5: DOWNLOAD ====================

with tab5:
    st.header("📥 Download Generated Files")
    
    if 'generated_code' in st.session_state:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.download_button(
                label="📄 Download Python Code",
                data=st.session_state.generated_code,
                file_name="adf_pipeline_generated.py",
                mime="text/plain"
            )
        
        with col2:
            if 'agent1_result' in st.session_state:
                st.download_button(
                    label="📊 Download Agent 1 Result",
                    data=json.dumps(st.session_state.agent1_result, indent=2),
                    file_name="agent1_csv_analysis.json",
                    mime="application/json"
                )
        
        with col3:
            if 'agent2_result' in st.session_state:
                st.download_button(
                    label="📋 Download Agent 2 Result",
                    data=json.dumps(st.session_state.agent2_result, indent=2),
                    file_name="agent2_datatype_mapping.json",
                    mime="application/json"
                )
        
        st.subheader("📦 Complete Package Download")
        if st.button("⬇️ Create Complete Package (ZIP)"):
            import zipfile
            import io
            
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                zip_file.writestr('adf_pipeline.py', st.session_state.generated_code)
                if 'agent1_result' in st.session_state:
                    zip_file.writestr('analysis_agent1.json', json.dumps(st.session_state.agent1_result, indent=2))
                if 'agent2_result' in st.session_state:
                    zip_file.writestr('analysis_agent2.json', json.dumps(st.session_state.agent2_result, indent=2))
            
            st.download_button(
                label="📦 Download Complete Package (ZIP)",
                data=zip_buffer.getvalue(),
                file_name="adf_pipeline_package.zip",
                mime="application/zip"
            )
    else:
        st.info("👈 Generate code first to download")

# ==================== FOOTER ====================

st.markdown("---")
st.markdown("""
    <div style='text-align: center'>
    <p>🤖 Azure OpenAI Agent System for ADF Pipeline Generation</p>
    <p><small>Powered by Azure OpenAI, Azure Data Factory, and Streamlit</small></p>
    </div>
    """, unsafe_allow_html=True)