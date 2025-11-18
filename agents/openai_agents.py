#!/usr/bin/env python3
"""
Azure OpenAI Agents for CSV Analysis, Data Type Detection, and Code Generation
Three-agent system: Agent 1 (CSV Analysis), Agent 2 (Data Type Detection), Agent 3 (Code Generation)
"""

import os
import json
import pandas as pd
from openai import AzureOpenAI
from dotenv import load_dotenv
import streamlit as st
import re
import traceback

load_dotenv()


class AzureOpenAIAgents:
    def __init__(self):
        """Initialize Azure OpenAI client with configuration from Streamlit secrets or environment variables"""
        api_key = None
        api_version = None
        azure_endpoint = None
        model = None
        
        try:
            if hasattr(st, 'secrets') and st.secrets:
                api_key = st.secrets.get('AZURE_OPENAI_KEY')
                api_version = st.secrets.get('AZURE_OPENAI_API_VERSION')
                azure_endpoint = st.secrets.get('AZURE_OPENAI_ENDPOINT')
                model = st.secrets.get('AZURE_OPENAI_DEPLOYMENT')
        except Exception:
            pass
        
        if not api_key:
            api_key = os.getenv('AZURE_OPENAI_KEY')
        if not api_version:
            api_version = os.getenv('AZURE_OPENAI_API_VERSION') or '2024-02-15-preview'
        if not azure_endpoint:
            azure_endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
        if not model:
            model = os.getenv('AZURE_OPENAI_DEPLOYMENT') or 'gpt-4'
        
        if not api_key:
            self.client = None
            self.model = None
            self.init_error = "OpenAI API key is not configured."
            print(self.init_error)
            return
        if not azure_endpoint:
            self.client = None
            self.model = None
            self.init_error = "OpenAI endpoint is not configured."
            print(self.init_error)
            return
        
        azure_endpoint = azure_endpoint.rstrip('/')
        
        # Initialize client with error handling
        try:
            self.client = AzureOpenAI(
                api_key=api_key,
                api_version=api_version,
                azure_endpoint=azure_endpoint
            )
            self.model = model
            self._sample_code_reference_cache = None
            self.init_error = None
            print(f"OpenAI client initialized with endpoint: {azure_endpoint}, model: {model}")
        except TypeError as e:
            # Handle version compatibility issues (like 'proxies' parameter)
            if 'proxies' in str(e) or 'unexpected keyword' in str(e):
                print(f"Warning: OpenAI client initialization issue: {e}. Attempting alternative initialization...")
                # Try with minimal parameters
                try:
                    self.client = AzureOpenAI(
                        api_key=api_key,
                        api_version=api_version,
                        azure_endpoint=azure_endpoint
                    )
                    self.model = model
                    self._sample_code_reference_cache = None
                    self.init_error = None
                    print(f"OpenAI client initialized successfully (alternative method)")
                except Exception as e2:
                    self.client = None
                    self.model = None
                    self.init_error = f"OpenAI client initialization failed: {str(e2)}"
                    print(self.init_error)
            else:
                self.client = None
                self.model = None
                self.init_error = f"OpenAI client initialization failed: {str(e)}"
                print(self.init_error)
        except Exception as e:
            self.client = None
            self.model = None
            self.init_error = f"OpenAI client initialization failed: {str(e)}"
            print(self.init_error)
    
    # ==================== Streaming Helper Methods ====================
    
    def _stream_chat_completion(self, messages, system_message=None, temperature=0.3, 
                                max_tokens=16000, stream_container=None, show_in_container=True,
                                response_format=None):
        """
        Stream chat completion response for real-time display in Streamlit.
        
        Args:
            messages: List of message dicts for the conversation
            system_message: Optional system message (will be prepended to messages)
            temperature: Sampling temperature (default: 0.3)
            max_tokens: Maximum tokens to generate (default: 16000)
            stream_container: Streamlit empty widget for displaying stream (optional)
            show_in_container: If True, display in container; if False, yield for st.write_stream()
            response_format: Optional response format (e.g., {"type": "json_object"})
        
        Returns:
            str: Complete response text (when show_in_container=True)
        """
        if self.client is None:
            raise ValueError("OpenAI client is not initialized")
        
        # Prepare messages with system message if provided
        if system_message:
            full_messages = [{"role": "system", "content": system_message}] + messages
        else:
            full_messages = messages
        
        # Prepare request parameters
        request_params = {
            "model": self.model,
            "messages": full_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True
        }
        
        # Add response format if provided
        if response_format:
            request_params["response_format"] = response_format
        
        try:
            # Create streaming request
            stream = self.client.chat.completions.create(**request_params)
            
            full_response = ""
            for chunk in stream:
                if chunk.choices and len(chunk.choices) > 0:
                    delta = chunk.choices[0].delta
                    if delta and delta.content is not None:
                        content = delta.content
                        full_response += content
                        
                        # Display in container if provided
                        if show_in_container and stream_container:
                            # Determine format based on content
                            if full_response.strip().startswith('{') or full_response.strip().startswith('['):
                                # JSON-like content
                                stream_container.markdown(f"```json\n{full_response}▌\n```")
                            elif '```' in full_response or 'def ' in full_response or 'import ' in full_response:
                                # Code-like content
                                stream_container.markdown(f"```python\n{full_response}▌\n```")
                            else:
                                # Plain text
                                stream_container.markdown(f"{full_response}▌")
            
            # Remove cursor and show final response
            if show_in_container and stream_container:
                if full_response.strip().startswith('{') or full_response.strip().startswith('['):
                    stream_container.markdown(f"```json\n{full_response}\n```")
                elif '```' in full_response or 'def ' in full_response or 'import ' in full_response:
                    stream_container.markdown(f"```python\n{full_response}\n```")
                else:
                    stream_container.markdown(full_response)
            
            return full_response
            
        except Exception as e:
            print(f"Error in streaming: {type(e).__name__}: {e}")
            traceback.print_exc()
            # Fallback to non-streaming mode
            try:
                request_params["stream"] = False
                if response_format:
                    request_params["response_format"] = response_format
                response = self.client.chat.completions.create(**request_params)
                full_response = response.choices[0].message.content
                if stream_container and show_in_container:
                    stream_container.markdown(f"⚠️ Streaming failed, using non-streaming mode\n\n{full_response}")
                return full_response
            except Exception as fallback_error:
                print(f"Fallback also failed: {fallback_error}")
                raise e
    
    # ==================== Prompt Constants ====================
    # Context-aware Agent 1 system guidance for robust domain/entity detection
    AGENT_1_CONTEXT_AWARE_PROMPT = (
        "You are a Data Warehouse Architect specializing in multi-domain data analysis.\n"
        "Identify domain (Healthcare, Sales, Finance, Automobile, Retail); classify columns into dimension keys, "
        "attributes, fact measures, and foreign keys. Ensure at least 3 dimensions and complete FK coverage.\n"
    )

    # Agent 3 dataflow rule to avoid duplication of groupBy columns in aggregate()
    AGENT_3_DYNAMIC_RESOURCE_PROMPT = (
        "In aggregate(groupBy(...)), groupBy columns are automatically in output and must NOT be duplicated in the "
        "aggregate list. Aggregate only non-groupBy columns with first/sum/avg/etc.\n"
    )
    
    # Agent 3 Complete System Prompt - 3-Layer Architecture Validation
    COMPLETE_AGENT_3_SYSTEM_PROMPT = """⚠️ CRITICAL PRIORITY INSTRUCTION ⚠️
═══════════════════════════════════════════════════════════════════════════
BEFORE generating code, mentally count the dimensions from Agent 1 output.
If dimension_count = 5, your dataflow script MUST have:
- 5 SelectDimXXX blocks (one for each dimension)
- 5 AggregateDimXXX blocks (one for each dimension) 
- OPTIONAL: Cast/Derive blocks based on Agent 2 data type recommendations
- 5 LoadDimXXX blocks (one for each dimension)
- 1 SelectFactXXX block
- OPTIONAL: Cast block for fact table based on Agent 2
- 1 LoadFactXXX block

MINIMUM TOTAL = 17 transformation blocks for 5 dimensions (Select + Aggregate + Load)
ACTUAL TOTAL = 17+ depending on CAST/DERIVE transformations added

If your generated script has < 10 transformation blocks, YOU STOPPED TOO EARLY!
If you only have 2 blocks (SelectFact + LoadFact), you MISSED ALL DIMENSIONS!
═══════════════════════════════════════════════════════════════════════════

You are an expert Azure Data Factory Python SDK code generator.

YOUR TASK: Generate COMPLETE Python code for ADF pipelines.

CRITICAL UNDERSTANDING:

════════════════════════════════════════════════════════════════════════
ADF Pipeline has 3 layers:
1. RESOURCE LAYER: resource_names, datasets, linked services
2. DATAFLOW SCRIPT LAYER: Transformation logic (source → select → aggregate → sink)
3. CONFIGURATION LAYER: Sinks, transformations registration

ALL 3 LAYERS MUST MATCH PERFECTLY!
════════════════════════════════════════════════════════════════════════

LAYER 1 VALIDATION: Resource Names
───────────────────────────────────
For each dimension from Agent 1:
✓ Must have entry in resource_names
✓ Must have dataset creation method
✓ Must have sink definition
Count Check: resources = static + dimensions + 1 fact

LAYER 2 VALIDATION: Dataflow Script
────────────────────────────────────
For EACH dimension from Agent 1:
✓ Must have: StagingSource select(...) ~> SelectDimX
✓ Must have: SelectDimX aggregate(...) ~> AggregateDimX
✓ OPTIONAL: Cast/Derive transformations between Aggregate and Sink
✓ Must have: Final transformation sink(...) ~> LoadDimX
Count Check:
- SELECT = dimension_count + 1 fact
- AGGREGATE = dimension_count
- CAST/DERIVE = Based on Agent 2 data types (may be 0 to many)
- LOAD = dimension_count + 1 fact

COLUMN COMPLETENESS VALIDATION (CRITICAL):
───────────────────────────────────────────
✓ Source CSV output MUST include ALL columns needed for ALL dimensions and fact table
✓ Each dimension's select MUST include ALL columns from Agent 1's dimension definition
  - Example: DimPatient MUST have ALL 18 columns listed in Agent 1
  - Example: DimDoctor MUST have ALL 9 columns listed in Agent 1
  - Example: DimHospital MUST have ALL 6 columns listed in Agent 1
✓ Fact table select MUST include ALL columns from Agent 1's fact_columns list
  - Example: FactVisit MUST have ALL 13 columns (Visit_ID, Visit_Date, Visit_Time, Discharge_Date, Billing_Date, Total_Amount, Insurance_Covered_Amount, Patient_Pay_Amount, Length_of_Stay_Days, Visit_Duration_Minutes, Procedure_Code, Diagnosis_Code, Invoice_ID)
✓ Use EXACT column names from Agent 2's datatype_mapping.json
✓ Column counts MUST match Agent 1/Agent 2 outputs exactly
✓ DO NOT omit any columns - every column in Agent 1's definitions MUST be included
✓ DO NOT add columns not in Agent 1/Agent 2 outputs

LAYER 3 VALIDATION: Sinks and Transformations
──────────────────────────────────────────────
For each transformation in script:
✓ Must have matching Transformation(name=...) in list
✓ Must have matching DataFlowSink(name=...) in sinks
Count Check:
- transformations list count = script transformation count
- sinks list count = script sink count

════════════════════════════════════════════════════════════════════════
GENERATION ALGORITHM (FOLLOW EXACTLY)
═════════════════════════════════════

STEP 1: Parse Agent 1 output
───────────────────────────
dimensions = agent1_output['dimensions']  # Dict of all dimensions
fact_table = agent1_output['fact_table']
dimension_count = len(dimensions)
VERIFY: You can see at least 3 dimensions. If not, STOP and ask for complete output.

STEP 2: Generate Layer 1 - Resource Names
──────────────────────────────────────────
return {{
    # STATIC - Copy exactly
    'sql_linked_service': 'SQLLinkedServiceConnection',
    'blob_linked_service': 'AzureBlobStorageConnection',
    'union_dataflow': 'UnionAll...CSVs',
    'transform_dataflow': 'TransformToFactDimension',
    'pipeline': '...CSVToSQLPipeline',
    
    # DYNAMIC - From Agent 1
    'fact_table_dataset': f'Fact{{fact_table_name}}Dataset',
    
    # FOR EACH DIMENSION - MUST LOOP THROUGH ALL
    FOR each dimension_name in dimensions:
        'dim_{{name}}_dataset': f'Dim{{name}}Dataset'
}}
VERIFY: Count = 6 static + 1 fact + dimension_count dimensions

STEP 3: Generate Layer 2 - Dataflow Script
────────────────────────────────────────────
script = \"\"\"source(...) ~> StagingSource

\"\"\"
# THIS LOOP MUST EXECUTE FOR EVERY DIMENSION
# DO NOT STOP EARLY, DO NOT SKIP ANY
FOR each dimension_name in sorted(dimensions.keys()):
    dimension = dimensions[dimension_name]
    primary_key = dimension['primary_key']
    columns = dimension['columns']
    
    # Generate SELECT
    script += f\"\"\"StagingSource select(mapColumn(
      {{',\\n      '.join(columns)}}
 ),
 skipDuplicateMapInputs: true,
 skipDuplicateMapOutputs: true) ~> Select{{dimension_name}}

\"\"\"
    
    # Generate AGGREGATE (WITHOUT duplicate PK!)
    other_columns = [c for c in columns if c != primary_key]
    agg_lines = []
    FOR each col in other_columns:
        agg_lines.append(f"{{col}} = first({{col}})")
    
    agg_expr = ',\\n     '.join(agg_lines)
    
    script += f\"\"\"Select{{dimension_name}} aggregate(groupBy({{primary_key}}),
     {{agg_expr}}) ~> Aggregate{{dimension_name}}

Aggregate{{dimension_name}} sink(allowSchemaDrift: true,
 validateSchema: false,
 deletable:false,
 insertable:true,
 updateable:false,
 upsertable:false,
 format: 'table',
 skipDuplicateMapInputs: true,
 skipDuplicateMapOutputs: true,
 errorHandlingOption: 'stopOnFirstError') ~> Load{{dimension_name}}

\"\"\"
# FACT TABLE (after dimension loop)
script += f\"\"\"StagingSource select(mapColumn(
      {{', '.join(fact_columns)}}
 )) ~> SelectFact
SelectFact sink(...) ~> LoadFact\"\"\"
VERIFY: 
- Count SELECT: Must equal dimension_count + 1
- Count AGGREGATE: Must equal dimension_count
- Count LOAD: Must equal dimension_count + 1

════════════════════════════════════════════════════════════════════════════════
CRITICAL INSTRUCTION: COMPLETE SCRIPT GENERATION (READ CAREFULLY!)
════════════════════════════════════════════════════════════════════════════════

PROBLEM: AI often stops generating the script early, creating only fact table
transformations and missing ALL dimension transformations.

MANDATORY SCRIPT STRUCTURE:
───────────────────────────

script = \"\"\"source(output(
      {{all_csv_columns}}
 ),
 allowSchemaDrift: true,
 validateSchema: false,
 ignoreNoFilesFound: false) ~> StagingSource

\"\"\"

# ════════════════════════════════════════════════════════════════════════════
# DIMENSION TRANSFORMATIONS LOOP - MUST EXECUTE FOR EVERY DIMENSION
# DO NOT SKIP THIS LOOP! DO NOT STOP EARLY!
# ════════════════════════════════════════════════════════════════════════════

dimensions = agent1_output['dimensions']  # Must have: DimDoctor, DimHospital, DimMedication, DimPatient, DimDate

FOR EACH dimension_name IN dimensions.keys():
    dimension = dimensions[dimension_name]
    primary_key = dimension['primary_key']
    columns = dimension['columns']
    
    script += f\"\"\"StagingSource select(mapColumn(
      {{',\\n      '.join(columns)}}
 ),
 skipDuplicateMapInputs: true,
 skipDuplicateMapOutputs: true) ~> Select{{dimension_name}}

\"\"\"
    
    other_columns = [col for col in columns if col != primary_key]
    agg_exprs = []
    for col in other_columns:
        agg_exprs.append(f"{{col}} = first({{col}})")
    
    agg_expr = ',\\n     '.join(agg_exprs)
    
    script += f\"\"\"Select{{dimension_name}} aggregate(groupBy({{primary_key}}),
     {{agg_expr}}) ~> Aggregate{{dimension_name}}

\"\"\"
    
    script += f\"\"\"Aggregate{{dimension_name}} sink(allowSchemaDrift: true,
 validateSchema: false,
 deletable:false,
 insertable:true,
 updateable:false,
 upsertable:false,
 format: 'table',
 skipDuplicateMapInputs: true,
 skipDuplicateMapOutputs: true,
 errorHandlingOption: 'stopOnFirstError') ~> Load{{dimension_name}}

\"\"\"

# ════════════════════════════════════════════════════════════════════════════
# FACT TABLE TRANSFORMATIONS - ONLY AFTER ALL DIMENSIONS
# ════════════════════════════════════════════════════════════════════════════

fact_columns = agent1_output['fact_columns']
fact_name = agent1_output['fact_table']['name']  # e.g., 'FactVisit'

script += f\"\"\"StagingSource select(mapColumn(
      {{',\\n      '.join(fact_columns)}}
 ),
 skipDuplicateMapInputs: true,
 skipDuplicateMapOutputs: true) ~> Select{{fact_name}}

Select{{fact_name}} sink(allowSchemaDrift: true,
 validateSchema: false,
 deletable:false,
 insertable:true,
 updateable:false,
 upsertable:false,
 format: 'table',
 skipDuplicateMapInputs: true,
 skipDuplicateMapOutputs: true,
 errorHandlingOption: 'stopOnFirstError') ~> Load{{fact_name}}\"\"\"

# ════════════════════════════════════════════════════════════════════════════
# VERIFICATION BEFORE RETURNING SCRIPT (MANDATORY!)
# ════════════════════════════════════════════════════════════════════════════

dimension_count = len(dimensions)

# Count transformations in generated script
select_count = script.count(' ~> Select')
aggregate_count = script.count(' ~> Aggregate')
load_count = script.count(' ~> Load')

# Expected counts
expected_select = dimension_count + 1      # All dimensions + fact
expected_aggregate = dimension_count       # Only dimensions (fact has no aggregate)
expected_load = dimension_count + 1        # All dimensions + fact

print(f"SCRIPT GENERATION VERIFICATION:")
print(f"  Dimensions: {{dimension_count}}")
print(f"  SELECT transformations: {{select_count}} (expected {{expected_select}})")
print(f"  AGGREGATE transformations: {{aggregate_count}} (expected {{expected_aggregate}})")
print(f"  LOAD transformations: {{load_count}} (expected {{expected_load}})")

# CRITICAL VALIDATION
validation_passed = True
errors = []

if select_count < expected_select:
    errors.append(f"Missing SELECT transformations: found {{select_count}}, expected {{expected_select}}")
    validation_passed = False

if aggregate_count < expected_aggregate:
    errors.append(f"Missing AGGREGATE transformations: found {{aggregate_count}}, expected {{expected_aggregate}}")
    validation_passed = False

if load_count < expected_load:
    errors.append(f"Missing LOAD transformations: found {{load_count}}, expected {{expected_load}}")
    validation_passed = False

# Verify each dimension by name
for dim_name in dimensions.keys():
    if f' ~> Select{{dim_name}}' not in script:
        errors.append(f"Missing Select{{dim_name}} in script")
        validation_passed = False
    if f' ~> Aggregate{{dim_name}}' not in script:
        errors.append(f"Missing Aggregate{{dim_name}} in script")
        validation_passed = False
    if f' ~> Load{{dim_name}}' not in script:
        errors.append(f"Missing Load{{dim_name}} in script")
        validation_passed = False

if not validation_passed:
    error_msg = "SCRIPT GENERATION FAILED VALIDATION:\\n" + "\\n".join(errors)
    print(f"❌ {{error_msg}}")
    raise ValueError(error_msg)
else:
    print(f"✅ Script validation passed! All {{dimension_count}} dimensions included.")

# Return the COMPLETE script
return script

════════════════════════════════════════════════════════════════════════════════
EXAMPLE OUTPUT FOR HEALTHCARE (5 dimensions) - WITH CAST/DERIVE:
════════════════════════════════════════════════════════════════════════════════

The script string must contain AT MINIMUM:

source(...) ~> StagingSource

StagingSource select(...) ~> SelectDimPatient
SelectDimPatient aggregate(...) ~> AggregateDimPatient
AggregateDimPatient cast(...) ~> CastDimPatient  # IF Agent 2 recommends casting
CastDimPatient sink(...) ~> LoadDimPatient

StagingSource select(...) ~> SelectDimDoctor
SelectDimDoctor aggregate(...) ~> AggregateDimDoctor
AggregateDimDoctor cast(...) ~> CastDimDoctor  # IF Agent 2 recommends casting
CastDimDoctor sink(...) ~> LoadDimDoctor

StagingSource select(...) ~> SelectDimHospital
SelectDimHospital aggregate(...) ~> AggregateDimHospital
AggregateDimHospital sink(...) ~> LoadDimHospital

StagingSource select(...) ~> SelectDimDate
SelectDimDate aggregate(...) ~> AggregateDimDate
AggregateDimDate derive(...) ~> DeriveDimDate  # For date conversions
DeriveDimDate sink(...) ~> LoadDimDate

StagingSource select(...) ~> SelectDimMedication
SelectDimMedication aggregate(...) ~> AggregateDimMedication
AggregateDimMedication sink(...) ~> LoadDimMedication

StagingSource select(...) ~> SelectFactVisit
SelectFactVisit cast(...) ~> CastFactVisit  # IF Agent 2 recommends casting measures
CastFactVisit sink(...) ~> LoadFactVisit

Note: Cast/Derive transformations are OPTIONAL and depend on Agent 2 recommendations.
Minimum: Select → Aggregate → Sink (17 blocks for 5 dimensions)
With Cast/Derive: Can be 20-30+ blocks depending on complexity

IF YOUR GENERATED SCRIPT IS < 100 LINES, YOU STOPPED TOO EARLY!
IF YOUR SCRIPT ONLY HAS SelectFactVisit + LoadFactVisit, YOU MISSED ALL DIMENSIONS!

════════════════════════════════════════════════════════════════════════════════
COMMON MISTAKES TO AVOID:
════════════════════════════════════════════════════════════════════════════════

❌ MISTAKE 1: Generating only fact table transformations
   → Your script has: SelectFactVisit, LoadFactVisit (2 transformations)
   → Missing: All 15 dimension transformations
   → FIX: Loop through ALL dimensions before adding fact

❌ MISTAKE 2: Listing transformations in array but not in script
   → transformations = [Transformation(name='SelectDimDoctor'), ...]
   → But script doesn't have: ~> SelectDimDoctor
   → FIX: Generate script first, then derive transformations list from script

❌ MISTAKE 3: Stopping generation due to token limits
   → Script cuts off after first dimension
   → FIX: Prioritize script generation, use concise variable names

❌ MISTAKE 4: Creating "Unknown" dimensions
   → ~> SelectDimUnknown (hallucinated)
   → FIX: Only use dimension names from Agent 1 output

❌ MISTAKE 5: Including primary_key in aggregate()
   → aggregate(groupBy(Doctor_ID), Doctor_ID = first(Doctor_ID), ...)
   → FIX: groupBy columns must NOT appear in aggregate list

❌ MISTAKE 6: Not adding CAST/DERIVE transformations when Agent 2 recommends them
   → Agent 2 says Age should be INT, but you skip the cast
   → FIX: Check Agent 2 data type recommendations and add cast/derive accordingly
   → Example: If Agent 2 recommends casting, add CastDimPatient between Aggregate and Sink

❌ MISTAKE 7: Omitting columns from dataflow scripts
   → Source CSV only has 5 columns instead of all dimension columns
   → DimPatient select only has 3 columns instead of all 18
   → Fact source missing Discharge_Date, Billing_Date, Diagnosis_Code, Invoice_ID
   → FIX: Include ALL columns from Agent 1's dimension definitions and fact_columns
   → FIX: Use exact column names from Agent 2's datatype_mapping.json
   → FIX: Verify column counts match Agent 1/Agent 2 outputs exactly
   → Example: DimPatient MUST include all 18 columns: Patient_ID, Patient_First_Name, Patient_Last_Name, Gender, DOB, Age, Marital_Status, Phone_Number, Email, Address, City, State, ZipCode, Ethnicity, Blood_Type, Allergies, Emergency_Contact_Name, Emergency_Contact_Phone

════════════════════════════════════════════════════════════════════════════════
MANDATORY COMPLETION CHECKLIST:
════════════════════════════════════════════════════════════════════════════════

Before returning the generated Python code, verify:

□ Script variable contains ALL dimension transformations (NOT just fact table!)
□ Script variable contains fact table transformations  
□ Script line count > 100 (for 5 dimensions)
□ Each dimension in Agent 1 has AT LEAST 3 blocks in script (Select, Aggregate, Load)
□ OPTIONAL: Cast/Derive transformations added based on Agent 2 recommendations
□ Transformations list includes ALL transformations (including Cast/Derive if added)
□ Sinks list matches all LoadX sinks in script
□ No "Unknown" or hallucinated dimensions
□ groupBy columns not duplicated in aggregate()
□ If Cast/Derive added, sink uses final transformation name (not Aggregate)
□ Syntax is valid (no trailing commas, proper indentation)
□ COLUMN VALIDATION: Dimension source includes ALL columns from ALL dimensions
□ COLUMN VALIDATION: Each dimension's select has ALL columns from Agent 1's definition
□ COLUMN VALIDATION: Fact source includes ALL columns from Agent 1's fact_columns
□ COLUMN VALIDATION: All columns from Agent 2's datatype_mapping.json are present
□ COLUMN VALIDATION: Column counts match Agent 1/Agent 2 outputs exactly
□ COLUMN VALIDATION: No columns are missing or omitted
□ COLUMN VALIDATION: Column names match exactly (case-sensitive)

IF ANY CHECKBOX IS UNCHECKED → REGENERATE THE CODE!

════════════════════════════════════════════════════════════════════════════════

TRANSFORMATION DECISION LOGIC (Context-Aware):
═══════════════════════════════════════════════

Based on Agent 2 data types, add appropriate transformations:

1. CAST TRANSFORMATIONS (When data types need conversion):
   - If Agent 2 suggests INT/BIGINT but pandas dtype is 'object' → Add cast
   - If Agent 2 suggests DECIMAL but pandas dtype is 'int64' → Add cast
   - If Agent 2 suggests DATETIME but pandas dtype is 'object' → Add cast
   
   Pattern:
   SelectDimX aggregate(groupBy(PK), ...) ~> AggregateDimX
   AggregateDimX cast(
       output(
           {{column_name}} as integer,
           {{another_col}} as decimal(18,2)
       ),
       errors: true) ~> CastDimX
   
   Then: CastDimX sink(...) instead of AggregateDimX sink(...)

2. DERIVE TRANSFORMATIONS (When calculated fields needed):
   Healthcare context:
   - Age calculation from DOB: year(currentDate()) - year(toDate(DOB))
   - Full name: concat(FirstName, ' ', LastName)
   - Date conversions: toDate(Visit_Date)
   
   Finance context:
   - Total amount: Quantity * UnitPrice
   - Tax: Amount * 0.18
   
   Automobile context:
   - Vehicle age: year(currentDate()) - ManufactureYear
   - Mileage category: case when Mileage < 50000 then 'Low' else 'High' end
   
   Pattern:
   SelectDimX aggregate(groupBy(PK), ...) ~> AggregateDimX
   AggregateDimX derive(
       Visit_Date = toDate(Visit_Date),
       Billing_Date = toDate(Billing_Date)
   ) ~> DeriveDimX
   
   Then: DeriveDimX sink(...) instead of AggregateDimX sink(...)

3. FILTER TRANSFORMATIONS (When data quality rules needed):
   - Remove null primary keys: filter(!isNull({{primary_key}}))
   - Remove invalid dates: filter(year({{date_col}}) >= 1900)
   - Remove negative amounts: filter({{amount_col}} >= 0)
   
   Pattern:
   SelectDimX filter(!isNull({{pk}}) && {{condition}}) ~> FilterDimX

4. SURROGATE KEY GENERATION (For dimensions without natural keys):
   Pattern:
   SelectDimX keyGenerate(
       output({{DimName}}_SK as long),
       startAt: 1L,
       stepValue: 1L
   ) ~> SurrogateKeyDimX

5. LOOKUP/JOIN (For foreign key resolution):
   When fact table has FKs that need SK mapping from dimensions
   Pattern:
   SelectFact lookup(
       SelectDimX@({{natural_key}}),
       SelectFact@({{fk_column}}) == SelectDimX@({{natural_key}}),
       multiple: false,
       pickup: 'any',
       broadcast: 'auto'
   ) ~> JoinDimX

CONTEXT-SPECIFIC PATTERNS:
─────────────────────────

Healthcare:
- DimDate: Always derive: Year, Month, Quarter, DayOfWeek from DateID
- DimPatient: Calculate Age from DOB, derive FullName
- DimDoctor: Derive FullName, YearsOfExperience from JoinDate
- FactVisit: Lookup all dimension SKs, calculate TotalCost

Finance:
- DimDate: Fiscal year calculation, Quarter mapping
- DimCustomer: Credit score categorization, customer segment
- FactTransaction: Calculate tax, net amount, profit margin

Automobile:
- DimVehicle: Calculate age, mileage category, depreciation
- DimCustomer: Age from DOB, location hierarchy (City→State→Country)
- FactSales: Calculate total price with taxes, discounts

Retail:
- DimProduct: Category hierarchy, price bands
- DimStore: Location hierarchy, store type categorization
- FactSales: Calculate line total, discounts, net sales

DECISION ALGORITHM:
──────────────────
FOR each dimension:
    0. CRITICAL FIRST STEP: Include ALL columns from Agent 1's dimension definition
       - Get exact column list from Agent 1's dimensions[DimName].columns
       - Get exact column list from Agent 2's datatype_mapping.json dimensions[DimName].columns
       - Verify column counts match exactly
       - Include EVERY column in source output and select transformation
    1. Check Agent 2 data types vs pandas dtypes → Add CAST if mismatch
    2. Check context (Healthcare/Finance/etc) → Add context-specific DERIVE
    3. Check for NULLs in sample data → Add FILTER if needed
    4. Check if natural key exists → Add SURROGATE KEY if missing
    5. Update transformation chain: Select → Cast → Derive → Filter → Aggregate → Sink

FOR fact table:
    0. CRITICAL FIRST STEP: Include ALL columns from Agent 1's fact_columns
       - Get exact column list from Agent 1's fact_columns
       - Get exact column list from Agent 2's datatype_mapping.json fact_table.fact_columns
       - Verify column counts match exactly
       - Include EVERY column in source output and select transformation
    1. Add CAST for measures
    2. Add LOOKUP for each FK to get dimension SKs
    3. Add DERIVE for calculated measures
    4. Sink to fact table

VALIDATION:
──────────
- Count transformations: Should be > (dimensions × 2) if context-aware logic applied
- Verify cast/derive appear in transformations list
- Verify script has proper chain: Select → Transform → Aggregate → Sink
- COLUMN VALIDATION: Verify ALL columns from Agent 1/Agent 2 are included
- COLUMN VALIDATION: Count columns in each select - must match Agent 1/Agent 2 exactly
- COLUMN VALIDATION: No columns missing, no extra columns added

STEP 4: Generate Layer 3 - Sinks List
──────────────────────────────────────
sinks = []
# FOR EACH DIMENSION - LOOP MUST COVER ALL
FOR each dimension_name in dimensions:
    clean_name = dimension_name.lower().replace('dim', '')
    dataset_key = f'dim_{{clean_name}}_dataset'
    
    sinks.append(
        DataFlowSink(
            name=f'Load{{dimension_name}}',
            dataset=DatasetReference(
                reference_name=self.names[dataset_key],
                type='DatasetReference'
            )
        )
    )
# Add fact sink
sinks.append(
    DataFlowSink(
        name=f'LoadFact',
        dataset=DatasetReference(
            reference_name=self.names['fact_table_dataset'],
            type='DatasetReference'
        )
    )
)
VERIFY: Count = dimension_count + 1

STEP 5: Generate Layer 3 - Transformations List
────────────────────────────────────────────────
transformations = []
# FOR EACH DIMENSION - LOOP MUST COVER ALL
FOR each dimension_name in dimensions:
    transformations.append(Transformation(name=f'Select{{dimension_name}}'))
    transformations.append(Transformation(name=f'Aggregate{{dimension_name}}'))
    # IF you added Cast/Derive for this dimension, add it to the list
    # transformations.append(Transformation(name=f'Cast{{dimension_name}}'))
    # transformations.append(Transformation(name=f'Derive{{dimension_name}}'))
# Add fact transformations
transformations.append(Transformation(name='SelectFact'))
# IF you added Cast for fact table, add it
# transformations.append(Transformation(name='CastFact'))
VERIFY: Count = (dimension_count × 2) + 1 + any Cast/Derive transformations added

STEP 6: FINAL VALIDATION BEFORE RETURNING
───────────────────────────────────────────
# Verify all 3 layers match
script_selects = count SELECT in script  # Must be dimension_count + 1
script_aggregates = count AGGREGATE in script  # Must be dimension_count
script_loads = count LOAD in script  # Must be dimension_count + 1
script_casts = count CAST in script  # Optional, depends on Agent 2
script_derives = count DERIVE in script  # Optional, depends on context

if script_selects != len(sinks):
    ERROR: "Layer mismatch! Script has {{script_selects}} SELECTs but sinks has {{len(sinks)}}"
    REGENERATE

# Count total transformations including Cast/Derive
script_total_transforms = script_selects + script_aggregates + script_casts + script_derives
if len(transformations) != script_total_transforms:
    ERROR: "Transformation count mismatch! Script has {{script_total_transforms}}, list has {{len(transformations)}}"
    REGENERATE

if any('Unknown' in script or 'Unknown' in dimension names):
    ERROR: "Hallucinated dimensions!"
    REGENERATE

# If script is too short, likely incomplete
if script length < 200 and dimension_count >= 3:
    ERROR: "Script too short! Missing transformations."
    REGENERATE

if validation passes:
    PRINT: "✓ All 3 layers validated and matching"
    RETURN: code
ELSE:
    PRINT: "Validation failed, regenerating..."
    REGENERATE

════════════════════════════════════════════════════════════════════════
CRITICAL RULES (DO NOT BREAK)
═════════════════════════════
□ LOOP MUST iterate through ALL dimensions
□ LOOP MUST NOT skip any dimension
□ LOOP MUST NOT create "Unknown" or hallucinated dimensions
□ Dataflow script MUST have transformations for ALL dimensions
□ Sinks list MUST match script sinks
□ Transformations list MUST match script transformations
□ All 3 layers MUST align perfectly

If any rule is broken, STOP and REGENERATE.
"""

    # Agent 3 Training Prompt - Comprehensive training-based code generation
    AGENT_3_TRAINING_PROMPT = """You are training to understand and generate Azure Data Factory Python SDK code.

IMPORTANT: You are learning from a SAMPLE CODE TEMPLATE.
Your task is NOT to recreate the sample, but to UNDERSTAND ITS PATTERN and apply it dynamically.

═════════════════════════════════════════════════════════════════════════════
STEP 1: UNDERSTAND THE SAMPLE CODE STRUCTURE
═════════════════════════════════════════════════════════════════════════════

SAMPLE CODE ANATOMY:

1. CLASS NAME STRUCTURE:
   Sample: class HospitalCSVToSQLPipeline
   Pattern: class {CONTEXT}CSVToSQLPipeline
   
   Where {CONTEXT} comes from Agent 1 domain detection:
   - Healthcare → HospitalCSVToSQLPipeline
   - Sales → SalesCSVToSQLPipeline
   - Finance → FinanceCSVToSQLPipeline
   - Automobile → AutomobileCSVToSQLPipeline
   
   RULE: Use Agent 1 detected domain context for class name

2. RESOURCE NAMING STRUCTURE:
   Sample has 3 categories:
   
   a) STATIC RESOURCES (Same for ALL contexts):
      - 'sql_linked_service': 'SQLLinkedServiceConnection{suffix}'
      - 'blob_linked_service': 'AzureBlobStorageConnection{suffix}'
      - 'transform_dataflow': 'TransformToFactDimension{suffix}'
      
      RULE: Copy these exactly, don't change
      
   b) CONTEXT-DEPENDENT RESOURCES (Change per domain):
      - 'source_csv_dataset': f'SourceXXXCSVDataset{suffix}'
      - 'staging_csv_dataset': f'StagingUnionXXXCSVDataset{suffix}'
      - 'union_dataflow': f'UnionAllXXXCSVs{suffix}'
      - 'pipeline': f'XXXCSVToSQLPipeline{suffix}'
      
      Where XXX = Domain from Agent 1
      - Healthcare → Source='SourceHealthcareCSVDataset', Union='UnionAllHealthcareCSVs'
      - Sales → Source='SourceSalesCSVDataset', Union='UnionAllSalesCSVs'
      
      RULE: Replace XXX with Agent 1 domain context
      
   c) DYNAMIC RESOURCES (Based on Agent 1 output):
      - 'fact_table_dataset': f'Fact{FactName}Dataset{suffix}'
      - 'dim_{name}_dataset': f'Dim{Name}Dataset{suffix}'
      
      Where:
      - FactName = From Agent 1['fact_table']['name'] (e.g., 'FactVisit', 'FactSales')
      - {name} = For each dimension in Agent 1['dimensions'] (e.g., 'patient', 'doctor')
      
3. DATAFLOW SCRIPT STRUCTURE:
   
   a) UNION DATAFLOW:
      - Source: All CSV columns from Agent 1
      - Sink: StagingSink
      - Pattern: source(output(...columns...), allowSchemaDrift: true, validateSchema: false, ignoreNoFilesFound: false) ~> SourceCSV
                 SourceCSV sink(...) ~> StagingSink
   
   b) TRANSFORM DATAFLOW - CRITICAL PATTERNS:
      
      ═══════════════════════════════════════════════════════════════════════════
      CRITICAL INSTRUCTION FOR DATAFLOW GENERATION
      ═══════════════════════════════════════════════════════════════════════════
      
      RULE 1: Parse Agent 1 Output
      ──────────────────────────────
      Extract ALL dimensions:
      agent1_dims = agent1_output['dimensions']
      Expected: {{'DimDoctor': {{...}}, 'DimDate': {{...}}, 'DimMedication': {{...}}, 
                 'DimPatient': {{...}}, 'DimHospital': {{...}}}}
      Dimension count = len(agent1_dims)  # Should be 5 or more
      
      VERIFY you can see ALL dimensions. If you see less than 3, STOP and say:
      "ERROR: Agent 1 output incomplete. Only seeing N dimensions, please provide all."
      
      RULE 2: Generate Dataflow Script Using EXPLICIT FOR LOOP
      ──────────────────────────────────────────────────────────
      
      STRUCTURE (Follow exactly):
      
      script = \"\"\"source(...) ~> StagingSource
      
      \"\"\"
      
      # THIS LOOP MUST EXECUTE FOR EVERY SINGLE DIMENSION
      # DO NOT SKIP ANY, DO NOT STOP EARLY
      for each_dimension in agent1_dims:
          dimension_name = each_dimension  # e.g., 'DimDoctor'
          dim_data = agent1_dims[dimension_name]
          primary_key = dim_data['primary_key']
          columns = dim_data['columns']
          
          # GENERATE SELECT
          script += f\"\"\"StagingSource select(mapColumn(
        {{columns joined by comma}}
       ),
       skipDuplicateMapInputs: true,
       skipDuplicateMapOutputs: true) ~> Select{{dimension_name}}
      
      \"\"\"
          
          # GENERATE AGGREGATE (exclude primary_key from aggregate list)
          other_cols = [c for c in columns if c != primary_key]
          agg_lines = []
          for col in other_cols:
              agg_lines.append(f"{{col}} = first({{col}})")
          
          script += f\"\"\"Select{{dimension_name}} aggregate(groupBy({{primary_key}}),
       {{agg_lines joined by comma}}) ~> Aggregate{{dimension_name}}
      
      Aggregate{{dimension_name}} sink(allowSchemaDrift: true,
       validateSchema: false,
       deletable:false,
       insertable:true,
       updateable:false,
       upsertable:false,
       format: 'table',
       skipDuplicateMapInputs: true,
       skipDuplicateMapOutputs: true,
       errorHandlingOption: 'stopOnFirstError') ~> Load{{dimension_name}}
      
      \"\"\"
      
      # FACT TABLE (after loop completes)
      script += \"\"\"StagingSource select(mapColumn(
        {{fact columns}}
       ),
       skipDuplicateMapInputs: true,
       skipDuplicateMapOutputs: true) ~> SelectFactVisit
      
      SelectFactVisit sink(...) ~> LoadFactVisit\"\"\"
      
      RULE 3: VERIFY Before Returning
      ────────────────────────────────
      
      Count SELECT: Should equal dimension_count + 1
      Count AGGREGATE: Should equal dimension_count
      Count LOAD: Should equal dimension_count + 1
      
      Example: If dimension_count = 5:
        SELECT: 6 (5 dimensions + 1 fact)
        AGGREGATE: 5 (only dimensions)
        LOAD: 6 (5 dimensions + 1 fact)
      
      If counts don't match:
        DO NOT RETURN CODE
        REGENERATE with the loop
        VERIFY counts again
      
      RULE 4: NEVER Create "Unknown" Dimensions
      ─────────────────────────────────────────
      
      ❌ DO NOT CREATE:
        SelectDimUnknown
        AggregateDimUnknown
        LoadDimUnknown
        
      These are HALLUCINATIONS. Every dimension must come from Agent 1.
      
      ✅ CREATE ONLY:
        SelectDimDoctor, SelectDimDate, SelectDimMedication, 
        SelectDimPatient, SelectDimHospital (from Agent 1)
      
      RULE 5: Exact Names Must Match
      ───────────────────────────────
      
      Dimension name from Agent 1: 'DimPatient'
      Transform names MUST be: 'SelectDimPatient', 'AggregateDimPatient', 'LoadDimPatient'
      
      Dimension name from Agent 1: 'DimDoctor'
      Transform names MUST be: 'SelectDimDoctor', 'AggregateDimDoctor', 'LoadDimDoctor'
      
      DO NOT CHANGE or shorten the dimension names.
      
      ═══════════════════════════════════════════════════════════════════════════
   
4. TRANSFORMATIONS ARRAY:
   Must include EVERY transformation from script:
   transformations=[
       Transformation(name='SelectDimPatient'),
       Transformation(name='AggregateDimPatient'),
       Transformation(name='CastDimPatient'),  # if cast exists
       Transformation(name='SelectDimDoctor'),
       Transformation(name='AggregateDimDoctor'),
       Transformation(name='CastDimDoctor'),  # if cast exists
       Transformation(name='SelectDimDate'),
       Transformation(name='AggregateDimDate'),
       Transformation(name='DeriveDimDate'),  # if derive exists
       Transformation(name='SelectFactVisit'),
       Transformation(name='CastFactVisit')
   ]

5. DATASET CREATION:
   - Fact: create_fact_table_dataset() - uses schema/table from destination_tables
   - Dimensions: create_dimension_datasets() - loops through all dimensions from Agent 1
   - Each dimension must have corresponding dataset_key in resource_names

═════════════════════════════════════════════════════════════════════════════
STEP 2: GENERATION ALGORITHM
═════════════════════════════════════════════════════════════════════════════

When generating code

1. Extract context from Agent 1: domain_context, dimensions, fact_table
2. Build resource_names dictionary dynamically
3. Generate class name: {Context}CSVToSQLPipeline
4. For transform_dataflow script: See STEP 4 below (CRITICAL)
5. Use destination_tables for actual schema.table names (not Agent 1 proposed names)

═════════════════════════════════════════════════════════════════════════════
STEP 4: GENERATE COMPLETE DATAFLOW SCRIPT (CRITICAL - READ CAREFULLY)
═════════════════════════════════════════════════════════════════════════════

The dataflow script MUST include transformations for ALL dimensions from Agent 1.

STRUCTURE (Follow EXACTLY):

script = \"\"\"source(output(...columns...), ...) ~> StagingSource

\"\"\"

dimensions = agent1_output['dimensions']
dimension_count = len(dimensions)

for dimension_name, dimension_data in dimensions.items():
    columns = dimension_data['columns']
    primary_key = dimension_data['primary_key']
    
    script += f\"\"\"StagingSource select(mapColumn(
      {{', '.join(columns)}}
 ),
 skipDuplicateMapInputs: true,
 skipDuplicateMapOutputs: true) ~> Select{{dimension_name}}

\"\"\"
    
    other_columns = [col for col in columns if col != primary_key]
    aggregate_list = [f"{{col}} = first({{col}})" for col in other_columns]
    
    script += f\"\"\"Select{{dimension_name}} aggregate(groupBy({{primary_key}}),
     {{', '.join(aggregate_list)}}) ~> Aggregate{{dimension_name}}

\"\"\"
    
    script += f\"\"\"Aggregate{{dimension_name}} sink(allowSchemaDrift: true,
 validateSchema: false,
 deletable:false,
 insertable:true,
 updateable:false,
 upsertable:false,
 format: 'table',
 skipDuplicateMapInputs: true,
 skipDuplicateMapOutputs: true,
 errorHandlingOption: 'stopOnFirstError') ~> Load{{dimension_name}}

\"\"\"

fact_columns = agent1_output.get('fact_columns', [])
fact_table_name = agent1_output['fact_table']['name'] if 'fact_table' in agent1_output else 'FactVisit'

script += f\"\"\"StagingSource select(mapColumn(
      {{', '.join(fact_columns)}}
 ),
 skipDuplicateMapInputs: true,
 skipDuplicateMapOutputs: true) ~> Select{{fact_table_name}}

Select{{fact_table_name}} sink(allowSchemaDrift: true,
 validateSchema: false,
 deletable:false,
 insertable:true,
 updateable:false,
 upsertable:false,
 format: 'table',
 skipDuplicateMapInputs: true,
 skipDuplicateMapOutputs: true,
 errorHandlingOption: 'stopOnFirstError') ~> Load{{fact_table_name}}\"\"\"

VERIFICATION BEFORE RETURNING (MANDATORY):

Count the following in the generated script string:
- Count SelectX in script = dimension_count + 1
  Example: If dimension_count = 5, must have 6 SelectX (5 dimensions + 1 fact)
  
- Count AggregateX in script = dimension_count
  Example: If dimension_count = 5, must have 5 AggregateX (only dimensions, no fact)
  
- Count LoadX in script = dimension_count + 1
  Example: If dimension_count = 5, must have 6 LoadX (5 dimensions + 1 fact)

FOR HOSPITAL/HEALTHCARE CONTEXT (5 dimensions):
  - SELECT: Must be >= 6 (5 dimensions + 1 fact)
  - AGGREGATE: Must be = 5 (only dimensions)
  - LOAD: Must be >= 6 (5 dimensions + 1 fact)

IF COUNTS DON'T MATCH:
  ❌ DO NOT RETURN THE CODE
  ❌ DO NOT SKIP THIS VERIFICATION
  ✅ REGENERATE the dataflow script
  ✅ VERIFY counts again
  ✅ Only return when counts match exactly

EXAMPLE FOR HOSPITAL (5 dimensions):
  ✓ SelectDimDate
  ✓ AggregateDimDate
  ✓ LoadDimDate
  ✓ SelectDimDoctor
  ✓ AggregateDimDoctor
  ✓ LoadDimDoctor
  ✓ SelectDimHospital
  ✓ AggregateDimHospital
  ✓ LoadDimHospital
  ✓ SelectDimMedication
  ✓ AggregateDimMedication
  ✓ LoadDimMedication
  ✓ SelectDimPatient
  ✓ AggregateDimPatient
  ✓ LoadDimPatient
  ✓ SelectFactVisit
  ✓ LoadFactVisit
  
  Total: 17 transformations (15 for dimensions + 2 for fact)

═════════════════════════════════════════════════════════════════════════════
STEP 3: VALIDATION CHECKLIST
═════════════════════════════════════════════════════════════════════════════

Before returning code, verify:
□ Class name matches context
□ Resource names include ALL dimensions from Agent 1
□ Transform dataflow script has blocks for ALL dimensions
□ Transformations array includes ALL transformation names
□ Sink names match LoadDimX / LoadFactY pattern
□ Dataset creation includes ALL dimensions
□ No hardcoded sample values (use Agent 1/2 outputs)
□ groupBy columns not duplicated in aggregate()

═════════════════════════════════════════════════════════════════════════════
REMEMBER: Understand the PATTERN, not copy the SAMPLE!
═════════════════════════════════════════════════════════════════════════════"""
    
    # ==================== AGENT 1: CSV ANALYSIS ====================
    
    def analyze_csv_structure(self, df, csv_filename):
        """Delegate to the safe v2 implementation"""
        return self.analyze_csv_structure_v2(df, csv_filename)
    
    def analyze_csv_structure_v2(self, df, csv_filename, target_tables=None, stream_container=None):
        """
        Safe version of Agent 1 CSV analysis that always returns a result.
        NEW: Compares CSV structure with target fact and dimension tables if provided.
        
        Args:
            df: DataFrame with CSV data
            csv_filename: Name of the CSV file
            target_tables: Dict with target table schemas {table_name: {column: {type, nullable}}}
            stream_container: Optional Streamlit container for displaying streaming response
        """
        if self.client is None:
            return self._create_fallback_analysis(df, csv_filename)
        try:
            columns = df.columns.tolist()
            shape = df.shape
            dtypes = {col: str(dt) for col, dt in df.dtypes.items()}
            sample = df.head(3).to_string()
            
            # Build target comparison context if provided
            target_context = ""
            if target_tables:
                # Validate target_tables is a dict
                if not isinstance(target_tables, dict):
                    print(f"Warning: target_tables is not a dict, got {type(target_tables)}")
                    target_tables = {}
                
                # Separate fact and dimension tables
                fact_targets = {}
                dim_targets = {}
                
                for table_name, table_info in target_tables.items():
                    # Validate table_name is a string
                    if not isinstance(table_name, str):
                        print(f"Warning: Skipping non-string table name: {type(table_name)} = {table_name}")
                        continue
                    
                    table_lower = table_name.lower()
                    if table_lower.startswith('fact') or table_lower.startswith('ft_'):
                        fact_targets[table_name] = table_info
                    elif table_lower.startswith('dim') or table_lower.startswith('dim_'):
                        dim_targets[table_name] = table_info
                
                target_context = f"""

╔═══════════════════════════════════════════════════════════════════════════════╗
║ CRITICAL: TARGET TABLES SELECTED IN UI                                        ║
║ You MUST match your output to these EXACT tables                              ║
╚═══════════════════════════════════════════════════════════════════════════════╝

These are the SPECIFIC tables the user selected in the Streamlit UI.
Your output MUST match these exact fact and dimension tables.

SELECTED FACT TABLE(S):"""
                
                for table_name, table_info in fact_targets.items():
                    # Validate table_info is a dict
                    if not isinstance(table_info, dict):
                        print(f"Warning: table_info is not a dict for {table_name}, got {type(table_info)}")
                        continue
                    
                    target_context += f"\n\n{table_name}:"
                    target_context += f"\n  Columns ({len(table_info)}):"
                    for col, col_info in table_info.items():
                        if isinstance(col_info, dict):
                            sql_type = col_info.get('type', 'UNKNOWN')
                            target_context += f"\n    - {col}: {sql_type}"
                        else:
                            target_context += f"\n    - {col}: {col_info}"
                
                target_context += "\n\nSELECTED DIMENSION TABLE(S):"
                
                for table_name, table_info in dim_targets.items():
                    # Validate table_info is a dict
                    if not isinstance(table_info, dict):
                        print(f"Warning: table_info is not a dict for {table_name}, got {type(table_info)}")
                        continue
                    
                    target_context += f"\n\n{table_name}:"
                    target_context += f"\n  Columns ({len(table_info)}):"
                    for col, col_info in table_info.items():
                        if isinstance(col_info, dict):
                            sql_type = col_info.get('type', 'UNKNOWN')
                            target_context += f"\n    - {col}: {sql_type}"
                        else:
                            target_context += f"\n    - {col}: {col_info}"
                
                dim_names = ', '.join(dim_targets.keys()) if dim_targets else 'NONE'
                fact_name = next(iter(fact_targets.keys()), 'NONE') if fact_targets else 'NONE'
                
                target_context += f"""

╔═══════════════════════════════════════════════════════════════════════════════╗
║ MANDATORY REQUIREMENTS:                                                       ║
╚═══════════════════════════════════════════════════════════════════════════════╝

1. Your output dimensions MUST be: {dim_names}
2. Your output fact table MUST be: {fact_name}
3. Map CSV columns to target table columns - use exact column names from targets
4. If CSV has extra columns not in targets, include them in appropriate table
5. If CSV is missing columns from targets, note it in reasoning
6. Dimension names MUST match target names exactly (case-sensitive)
7. DO NOT suggest different table names - use the target table names provided above

CRITICAL: Your JSON output must have:
- "dimensions": {{"DimX": {{"columns": [...], "primary_key": "..."}}, "DimY": {{...}}}}
- "fact_table": {{"name": "FactX", ...}}
- Match the table names shown in TARGET TABLES above EXACTLY
"""
            
            prompt = (
                self.AGENT_1_CONTEXT_AWARE_PROMPT + "\n\n" +
                "Analyze this CSV and propose fact/dimension split as JSON with keys: "
                "fact_columns, dimensions (with columns, primary_key), foreign_keys, reasoning.\n\n"
                f"CSV: {csv_filename} Rows={shape[0]} Cols={shape[1]}\n"
                f"Dtypes: {json.dumps(dtypes, indent=2)}\n\nSample:\n{sample}\n"
                + target_context
            )
            
            system_message = "You are a data warehouse architect expert. You compare source CSV structures with target database schemas."
            messages = [{"role": "user", "content": prompt}]
            
            # Use streaming if stream_container is provided
            if stream_container:
                try:
                    text = self._stream_chat_completion(
                        messages=messages,
                        system_message=system_message,
                        temperature=0.3,
                        max_tokens=16000,
                        stream_container=stream_container,
                        show_in_container=True
                    )
                except Exception as stream_error:
                    print(f"Streaming failed, falling back to non-streaming: {stream_error}")
                    # Fallback to non-streaming
                    resp = self.client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": system_message},
                            {"role": "user", "content": prompt},
                        ],
                        temperature=0.3,
                        max_tokens=16000,
                    )
                    text = resp.choices[0].message.content
            else:
                # Non-streaming mode
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.3,
                    max_tokens=16000,
                )
                text = resp.choices[0].message.content
            
            # Parse JSON response
            m = re.search(r"\{[\s\S]*\}", text)
            
            if m:
                try:
                    data = json.loads(m.group())
                    if isinstance(data, dict) and (data.get('fact_columns') or data.get('dimensions')):
                        return data
                except Exception:
                    pass
            
            return self._create_fallback_analysis(df, csv_filename)
        except Exception as e:
            print(f"Error in analyze_csv_structure_v2: {type(e).__name__}: {e}")
            traceback.print_exc()
            return self._create_fallback_analysis(df, csv_filename)
    
    def _create_fallback_analysis(self, df, csv_filename):
        """Create a basic fallback analysis when AI response parsing fails"""
        columns = df.columns.tolist()
        id_columns = [col for col in columns if 'id' in col.lower() and col.lower().endswith('_id')]
        numeric_columns = [col for col in df.select_dtypes(include=['number']).columns.tolist() if col not in id_columns]
        text_columns = df.select_dtypes(include=['object']).columns.tolist()

        fact_columns = []
        dimensions = {}
        fact_columns.extend(numeric_columns)

        dimension_keywords = {
            'DimPatient': ['patient'],
            'DimDoctor': ['doctor', 'physician'],
            'DimHospital': ['hospital', 'clinic', 'facility'],
            'DimDate': ['date', 'time'],
            'DimMedication': ['medication', 'drug', 'medicine'],
            'DimLocation': ['location', 'address', 'city', 'state', 'zip'],
            'DimDepartment': ['department', 'division']
        }

        for dim_name, keywords in dimension_keywords.items():
            matching_cols = [col for col in text_columns if any(k in col.lower() for k in keywords)]
            if matching_cols:
                pk_candidates = [col for col in id_columns if any(k in col.lower() for k in keywords)]
                primary_key = pk_candidates[0] if pk_candidates else matching_cols[0]
                dimensions[dim_name] = {'columns': matching_cols, 'primary_key': primary_key}

        if not dimensions and id_columns:
            for id_col in id_columns[:5]:
                dim_name = f"Dim{id_col.replace('_ID', '').replace('_', '').title()}"
                related_cols = [col for col in text_columns if id_col.replace('_ID', '').lower() in col.lower()]
                if related_cols:
                    dimensions[dim_name] = {'columns': [id_col] + related_cols[:10], 'primary_key': id_col}

        foreign_keys = {}
        for dim_name, dim_info in dimensions.items():
            pk = dim_info.get('primary_key')
            if pk in columns:
                foreign_keys[pk] = dim_name

        remaining_cols = [col for col in columns if col not in fact_columns and not any(col in d.get('columns', []) for d in dimensions.values())]
        fact_columns.extend(remaining_cols)

        if not fact_columns:
            fact_columns = numeric_columns[:20]

        if not dimensions and id_columns:
            first_id = id_columns[0]
            dim_name = f"Dim{first_id.replace('_ID', '').replace('_', '').title()}"
            related_cols = [col for col in text_columns[:10]]
            dimensions[dim_name] = {'columns': [first_id] + related_cols, 'primary_key': first_id}
            foreign_keys[first_id] = dim_name

        result = {
            'fact_columns': fact_columns[:50] if fact_columns else columns[:10],
            'dimensions': dimensions if dimensions else {},
            'foreign_keys': foreign_keys,
            'reasoning': f'Fallback analysis created using heuristics. Analyzed {len(columns)} columns from CSV file: {csv_filename}'
        }
        print(f"Fallback analysis created: {len(result['fact_columns'])} fact columns, {len(result['dimensions'])} dimensions")
        return result

    # ==================== HELPER FUNCTIONS ====================
    
    def _safe_json_loads(self, json_string):
        """Safely parse JSON string, returns None if parsing fails"""
        if not json_string or not isinstance(json_string, str):
            return None
        try:
            return json.loads(json_string)
        except (json.JSONDecodeError, ValueError, TypeError):
            return None
        except Exception:
            return None
    
    # ==================== AGENT 2: DATA TYPE DETECTION ====================
    
    def detect_column_datatypes(self, df, agent1_analysis=None, target_tables=None, stream_container=None):
        """
        Agent 2: Analyze column content and suggest appropriate SQL data types
        Uses Agent 1's fact/dimension analysis to make better recommendations
        NEW: Compares with target table datatypes if provided
        
        Args:
            df: DataFrame with CSV data
            agent1_analysis: Output from Agent 1 with fact/dimension structure
            target_tables: Dict with target table schemas {table_name: {column: {type, nullable}}}
            stream_container: Optional Streamlit container for displaying streaming response
        """
        try:
            if self.client is None:
                print("OpenAI client not available, using fallback data type detection...")
                return self._create_fallback_datatypes(df, agent1_analysis)
            
            # Build column samples safely
            column_samples = {}
            for col in df.columns:
                try:
                    null_count = int(df[col].isnull().sum())
                    unique_count = int(df[col].nunique())
                    sample_values = df[col].astype(str).head(10).tolist()
                    sample_values = [str(v) if v is not None else '' for v in sample_values]
                    column_samples[col] = {
                        "sample_values": sample_values,
                        "null_count": null_count,
                        "unique_count": unique_count,
                        "detected_type": str(df[col].dtype)
                    }
                except Exception:
                    column_samples[col] = {
                        "sample_values": [],
                        "null_count": 0,
                        "unique_count": 0,
                        "detected_type": "unknown"
                    }
            
            # Serialize to JSON safely
            try:
                column_samples_json = json.dumps(column_samples, indent=2, ensure_ascii=False, default=str)
                if not column_samples_json or len(column_samples_json) == 0:
                    raise ValueError("Empty JSON string")
            except Exception as json_err:
                print(f"JSON serialization failed: {json_err}, using fallback...")
                return self._create_fallback_datatypes(df, agent1_analysis)
            
            # Include Agent 1 analysis in prompt if available
            agent1_context = ""
            if agent1_analysis:
                try:
                    agent1_json = json.dumps(agent1_analysis, indent=2, ensure_ascii=False, default=str)
                    agent1_context = f"""
                    
Context from Agent 1 (Fact/Dimension Analysis):
{agent1_json}

Use this context to:
- Fact table columns typically need numeric types (INT, DECIMAL, FLOAT)
- Dimension table columns are usually descriptive (VARCHAR, NVARCHAR, DATE)
- Primary keys should be INT or BIGINT
- Foreign keys should match their referenced primary keys
                    """
                except Exception:
                    agent1_context = ""
            
            # Build target table context if provided
            target_context = ""
            if target_tables:
                # Validate target_tables is a dict
                if not isinstance(target_tables, dict):
                    print(f"Warning: target_tables is not a dict, got {type(target_tables)}")
                    target_tables = {}
                
                # Separate fact and dimension tables
                fact_targets = {}
                dim_targets = {}
                
                for table_name, table_info in target_tables.items():
                    # Validate table_name is a string
                    if not isinstance(table_name, str):
                        print(f"Warning: Skipping non-string table name: {type(table_name)} = {table_name}")
                        continue
                    
                    table_lower = table_name.lower()
                    if table_lower.startswith('fact') or table_lower.startswith('ft_'):
                        fact_targets[table_name] = table_info
                    elif table_lower.startswith('dim') or table_lower.startswith('dim_'):
                        dim_targets[table_name] = table_info
                
                target_context = f"""

╔═══════════════════════════════════════════════════════════════════════════════╗
║ CRITICAL: TARGET TABLE DATATYPES FROM SELECTED DATABASE                       ║
║ You MUST match suggested datatypes EXACTLY to target tables                   ║
╚═══════════════════════════════════════════════════════════════════════════════╝

These are the datatypes from tables selected in the Streamlit UI.
For each CSV column mapping to a target column, you MUST use the EXACT target datatype.

TARGET FACT TABLE DATATYPES:"""
                
                for table_name, table_info in fact_targets.items():
                    # Validate table_info is a dict
                    if not isinstance(table_info, dict):
                        print(f"Warning: table_info is not a dict for {table_name}, got {type(table_info)}")
                        continue
                    
                    target_context += f"\n\n{table_name}:"
                    for col, col_info in table_info.items():
                        if isinstance(col_info, dict):
                            sql_type = col_info.get('type', 'UNKNOWN')
                            nullable = 'NULL' if col_info.get('nullable', True) else 'NOT NULL'
                            target_context += f"\n    - {col}: {sql_type} ({nullable})"
                        else:
                            target_context += f"\n    - {col}: {col_info}"
                
                target_context += f"\n\nTARGET DIMENSION TABLE DATATYPES:"
                
                for table_name, table_info in dim_targets.items():
                    # Validate table_info is a dict
                    if not isinstance(table_info, dict):
                        print(f"Warning: table_info is not a dict for {table_name}, got {type(table_info)}")
                        continue
                    
                    target_context += f"\n\n{table_name}:"
                    for col, col_info in table_info.items():
                        if isinstance(col_info, dict):
                            sql_type = col_info.get('type', 'UNKNOWN')
                            nullable = 'NULL' if col_info.get('nullable', True) else 'NOT NULL'
                            target_context += f"\n    - {col}: {sql_type} ({nullable})"
                        else:
                            target_context += f"\n    - {col}: {col_info}"
                
                target_context += f"""

╔═══════════════════════════════════════════════════════════════════════════════╗
║ MANDATORY DATATYPE MATCHING RULES:                                            ║
╚═══════════════════════════════════════════════════════════════════════════════╝

1. If CSV column name matches target table column name EXACTLY, use target's datatype
2. Example: Target has 'CustomerName VARCHAR(100)' → suggest 'VARCHAR(100)', NOT NVARCHAR
3. Example: Target has 'Price DECIMAL(10,2)' → suggest 'DECIMAL(10,2)', NOT FLOAT or MONEY
4. Example: Target has 'OrderDate DATE' → suggest 'DATE', NOT DATETIME
5. For columns not in targets, analyze and suggest appropriate datatype
6. NO variations or "similar" types - use EXACT match from target tables
7. Case-sensitive column name matching

CRITICAL: Your output must contain datatypes that EXACTLY match target tables where columns map.
"""
            
            # Create prompt
            prompt = f"""Based on the following column data samples, suggest the most appropriate data types for Azure Data Factory (ADF) Data Flow.

{column_samples_json}
{agent1_context}
{target_context}

╔═══════════════════════════════════════════════════════════════════════════════╗
║ CRITICAL: ADF DATA FLOW TYPE REQUIREMENTS                                      ║
╚═══════════════════════════════════════════════════════════════════════════════╝

ADF Data Flow DSL ONLY supports these basic types (NO SQL-specific types):
- string (for all text: VARCHAR, NVARCHAR, CHAR, NCHAR, TEXT, NTEXT)
- integer (for INT, SMALLINT, TINYINT)
- long (for BIGINT)
- double (for FLOAT, REAL)
- decimal (for DECIMAL, NUMERIC, MONEY)
- boolean (for BIT)
- timestamp (for DATETIME, DATETIME2, SMALLDATETIME)
- date (for DATE only)
- byte (for BINARY, VARBINARY)
- binary (for IMAGE, VARBINARY(MAX))

SQL TYPE TO ADF TYPE MAPPING:
- VARCHAR, NVARCHAR, CHAR, NCHAR, TEXT, NTEXT → string
- INT, SMALLINT, TINYINT → integer
- BIGINT → long
- FLOAT, REAL → double
- DECIMAL(p,s), NUMERIC(p,s), MONEY → decimal (keep precision: decimal(18,2))
- BIT → boolean
- DATETIME, DATETIME2, SMALLDATETIME → timestamp
- DATE → date
- BINARY, VARBINARY → byte
- IMAGE, VARBINARY(MAX) → binary

CRITICAL RULES:
1. For cast operations in ADF, use ONLY: string, integer, long, double, decimal, boolean, timestamp, date
2. DO NOT use SQL types like: nvarchar, varchar, datetime2, etc.
3. For decimal types, use format: decimal(18,2) - ADF supports decimal with precision
4. For text columns, ALWAYS use: string (regardless of SQL type being VARCHAR or NVARCHAR)
5. For date columns, use: date (for DATE) or timestamp (for DATETIME/DATETIME2)
6. For numeric columns, use: integer, long, double, or decimal based on precision needs

For each column, consider:
- The actual data content (numbers, dates, text length, etc.)
- Null values and how to handle them
- Uniqueness (could be primary key?)
- Range and precision needed
- Whether column is in fact table (measures/metrics) or dimension table (descriptive attributes)
- EXACT MATCH with target table datatypes if mapping exists (but convert to ADF types)

Response conditions:
- Exclude columns from the CSV file in the output that are not in the target table.

Response format (JSON):
{{
    "columns": {{
        "column_name": {{
            "sql_type": "NVARCHAR(255)",
            "adf_type": "string",
            "nullable": true,
            "reasoning": "explanation"
        }}
    }}
}}

IMPORTANT: You MUST provide BOTH sql_type (for reference) AND adf_type (for ADF code generation).
The adf_type MUST be one of: string, integer, long, double, decimal, boolean, timestamp, date, byte, binary

Examples:
- SQL: NVARCHAR(255) → ADF: string
- SQL: DECIMAL(18,2) → ADF: decimal(18,2)
- SQL: INT → ADF: integer
- SQL: DATETIME2 → ADF: timestamp
- SQL: DATE → ADF: date
        """
        
            # Call OpenAI API
            system_message = """You are a database schema designer specializing in Azure Data Factory (ADF) Data Flow type mapping. 
You analyze data and recommend both SQL data types (for reference) and ADF-compatible types (for code generation).

CRITICAL ADF TYPE REQUIREMENTS:
- ADF Data Flow DSL ONLY supports: string, integer, long, double, decimal, boolean, timestamp, date, byte, binary
- NEVER use SQL-specific types like nvarchar, varchar, datetime2 in ADF cast operations
- Always provide BOTH sql_type (SQL Server type) AND adf_type (ADF-compatible type) in your output
- Convert SQL types to ADF types: VARCHAR/NVARCHAR → string, INT → integer, DECIMAL → decimal(18,2), DATETIME2 → timestamp, DATE → date

When target table datatypes are provided, you MUST:
1. Use the SQL type from target as sql_type
2. Convert it to ADF-compatible type as adf_type
3. Exclude columns in the output that are not in the target table"""
            
            messages = [{"role": "user", "content": prompt}]
            
            # Use streaming if stream_container is provided
            if stream_container:
                try:
                    response_text = self._stream_chat_completion(
                        messages=messages,
                        system_message=system_message,
                        temperature=0.3,
                        max_tokens=16000,
                        stream_container=stream_container,
                        show_in_container=True
                    )
                except Exception as stream_error:
                    print(f"Streaming failed, falling back to non-streaming: {stream_error}")
                    # Fallback to non-streaming
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": system_message},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0.3,
                        max_tokens=16000
                    )
                    response_text = response.choices[0].message.content
            else:
                # Non-streaming mode
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3,
                    max_tokens=16000
                )
                response_text = response.choices[0].message.content
            
            if not response_text:
                raise ValueError("Empty response from API")
            
            # Parse JSON response
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            
            if json_match:
                json_str = json_match.group()
                result = self._safe_json_loads(json_str)
                if result and isinstance(result, dict) and 'columns' in result:
                    return result
            
            print("JSON parsing failed, using fallback...")
            return self._create_fallback_datatypes(df, agent1_analysis)
                
        except Exception as e:
            print(f"Error in Agent 2: {type(e).__name__}: {e}")
            print(traceback.format_exc())
            return self._create_fallback_datatypes(df, agent1_analysis)
    
    def _create_fallback_datatypes(self, df, agent1_analysis=None):
        """Create fallback SQL data type mappings based on pandas dtypes"""
        try:
            columns = {}
            
            # Extract fact/dimension information from Agent 1
            fact_columns = []
            dimension_columns = {}
            if agent1_analysis:
                try:
                    fact_columns = agent1_analysis.get('fact_columns', [])
                    dimensions = agent1_analysis.get('dimensions', {})
                    for dim_name, dim_info in dimensions.items():
                        dim_cols = dim_info.get('columns', [])
                        for col in dim_cols:
                            dimension_columns[col] = dim_name
                except Exception:
                    pass
            
            for col in df.columns:
                try:
                    dtype = str(df[col].dtype)
                    null_count = int(df[col].isnull().sum())
                    max_length = 0
                    
                    is_fact_column = col in fact_columns
                    is_dimension_column = col in dimension_columns
                    
                    # Determine SQL type and ADF type based on pandas dtype
                    if 'int' in dtype:
                        try:
                            if df[col].notna().any():
                                max_val = float(df[col].max())
                                min_val = float(df[col].min())
                                if abs(max_val) > 2147483647 or abs(min_val) > 2147483647:
                                    sql_type = "BIGINT"
                                    adf_type = "long"
                                else:
                                    sql_type = "INT"
                                    adf_type = "integer"
                            else:
                                sql_type = "INT"
                                adf_type = "integer"
                        except Exception:
                            sql_type = "INT"
                            adf_type = "integer"
                    elif 'float' in dtype:
                        sql_type = "DECIMAL(18,2)"
                        adf_type = "decimal(18,2)"
                    elif 'datetime' in dtype or 'date' in dtype:
                        sql_type = "DATETIME"
                        adf_type = "timestamp"
                    elif 'bool' in dtype:
                        sql_type = "BIT"
                        adf_type = "boolean"
                    else:
                        # String type
                        try:
                            if df[col].notna().any():
                                max_length = int(df[col].astype(str).str.len().max())
                                if max_length > 4000:
                                    sql_type = "NVARCHAR(MAX)"
                                elif max_length > 255:
                                    sql_type = f"NVARCHAR({min(max_length * 2, 4000)})"
                                else:
                                    sql_type = f"NVARCHAR({max(max_length * 2, 50)})"
                            else:
                                sql_type = "NVARCHAR(255)"
                            adf_type = "string"
                        except Exception:
                            sql_type = "NVARCHAR(255)"
                            adf_type = "string"
                    
                    # Enhanced reasoning
                    reasoning = f"Fallback: Detected from pandas dtype: {dtype}, nulls: {null_count}"
                    if is_fact_column:
                        reasoning += ", fact table column"
                    elif is_dimension_column:
                        reasoning += f", dimension table ({dimension_columns[col]})"
                    if max_length > 0:
                        reasoning += f", max_length: {max_length}"
                    
                    columns[col] = {
                        "sql_type": sql_type,
                        "adf_type": adf_type,
                        "nullable": null_count > 0,
                        "reasoning": reasoning
                    }
                except Exception:
                    columns[col] = {
                        "sql_type": "NVARCHAR(255)",
                        "adf_type": "string",
                        "nullable": True,
                        "reasoning": "Fallback: Default type"
                    }
            
            result = {"columns": columns}
            print(f"Fallback data type detection created for {len(columns)} columns")
            return result
        except Exception as e:
            print(f"Error in fallback function: {e}")
            return {
                "columns": {
                    col: {
                        "sql_type": "NVARCHAR(255)",
                        "adf_type": "string",
                        "nullable": True,
                        "reasoning": "Fallback: Minimal safe default"
                    } for col in df.columns
                }
            }
    
    # ==================== AGENT 3: CODE GENERATION ====================
    
    def _normalize_dimensions(self, dimensions):
        """Normalize Agent 1 dimensions to a dict: {DimName: {columns:[], primary_key:''}}"""
        if isinstance(dimensions, dict):
            normalized = {}
            for name, info in dimensions.items():
                # Validate name is a string
                if not isinstance(name, str):
                    print(f"Warning: Skipping non-string dimension name: {name}")
                    continue
                
                dim_name = name if name.lower().startswith('dim') else f"Dim{name}"
                cols = []
                pk = ''
                if isinstance(info, dict):
                    cols = info.get('columns') or []
                    if isinstance(cols, dict):
                        cols = list(cols.keys())
                    if not isinstance(cols, list):
                        cols = []
                    pk = info.get('primary_key') or info.get('pk') or ''
                    if not pk and cols:
                        pk = cols[0]
                normalized[dim_name] = {"columns": cols, "primary_key": pk}
            return normalized
        if isinstance(dimensions, list):
            normalized = {}
            for item in dimensions:
                if isinstance(item, str):
                    dim_name = item if item.lower().startswith('dim') else f"Dim{item}"
                    normalized[dim_name] = {"columns": [], "primary_key": ''}
                elif isinstance(item, dict):
                    name = item.get('name') or item.get('dimension')
                    if not name and len(item) == 1:
                        name = next(iter(item.keys()))
                        maybe = item[name]
                        cols = maybe.get('columns') if isinstance(maybe, dict) else []
                        pk = maybe.get('primary_key') if isinstance(maybe, dict) else ''
                    else:
                        cols = item.get('columns') or []
                        pk = item.get('primary_key') or ''
                    
                    # Validate name is a string before calling .lower()
                    if not isinstance(name, str):
                        name = 'Unknown'
                    
                    dim_name = name if name and name.lower().startswith('dim') else f"Dim{name or 'Unknown'}"
                    if isinstance(cols, dict):
                        cols = list(cols.keys())
                    if not isinstance(cols, list):
                        cols = []
                    if not pk and cols:
                        pk = cols[0]
                    normalized[dim_name] = {"columns": cols, "primary_key": pk}
            return normalized
        return {}

    def generate_pipeline_prompt(self, csv_analysis, datatype_analysis, destination_tables, azure_config, 
                                 csv_data=None, blob_container=None, blob_folder=None, validation_feedback=None):
        """
        Agent 3A: Generate a comprehensive prompt that can be easily understood by Agent 3B.
        This creates a detailed instruction set for code generation.
        
        Args:
            validation_feedback: Optional feedback from Agent 3C validation to address issues
        """
        try:
            if csv_analysis is None:
                raise ValueError("CSV analysis (Agent 1 output) is required")
            if datatype_analysis is None:
                raise ValueError("Data type analysis (Agent 2 output) is required")
            if not destination_tables:
                raise ValueError("At least one destination table must be selected")
            
            if self.client is None:
                # Fallback to direct code generation if no OpenAI client
                return None
            
            csv_columns = csv_data.columns.tolist() if csv_data is not None else []
            
            fact_columns = csv_analysis.get('fact_columns', [])
            raw_dimensions = csv_analysis.get('dimensions', {})
            dimensions = self._normalize_dimensions(raw_dimensions)
            foreign_keys = csv_analysis.get('foreign_keys', {})
            
            column_types = {}
            if datatype_analysis and 'columns' in datatype_analysis:
                column_types = datatype_analysis['columns']
            
            fact_tables = []
            dim_tables = []
            table_schemas = {}
            for table_key, table_info in destination_tables.items():
                if '.' in table_key:
                    schema, table = table_key.split('.', 1)
                    table_schemas[table] = schema
                    tl = table.lower()
                    if tl.startswith('fact') or tl.startswith('ft_'):
                        fact_tables.append((table, schema))
                    elif tl.startswith('dim') or tl.startswith('dim_'):
                        dim_tables.append((table, schema))
                    else:
                        matched = False
                        for dim_name in dimensions.keys():
                            # Validate dim_name is a string
                            if not isinstance(dim_name, str):
                                continue
                            if dim_name.replace('Dim', '').lower() in tl:
                                dim_tables.append((table, schema))
                                matched = True
                                break
                        if not matched:
                            fact_tables.append((table, schema))
            
            # Prepare context for Agent 3A
            prompt_context = {
                'csv_columns': csv_columns,
                'fact_columns': fact_columns,
                'dimensions': dimensions,
                'foreign_keys': foreign_keys,
                'column_types': column_types,
                'fact_tables': fact_tables,
                'dim_tables': dim_tables,
                'table_schemas': table_schemas,
                'azure_config': azure_config,
                'blob_container': blob_container or 'applicationdata',
                'blob_folder': blob_folder or 'source'
            }
            
            # Build validation feedback section separately to avoid nested f-string issues
            validation_section = ""
            if validation_feedback:
                validation_section = f"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║ VALIDATION FEEDBACK FROM AGENT 3C (MUST ADDRESS)                             ║
╚═══════════════════════════════════════════════════════════════════════════════╝

The previous code generation had the following issues that MUST be fixed:

{validation_feedback}

CRITICAL: You MUST address ALL issues listed above in your decision JSON.
- If columns were missing, ensure ALL columns are included in column_mappings
- If transformations were missing, ensure activities array includes them
- If column counts were wrong, verify against Agent 1's exact column lists
- Review each issue carefully and ensure your decision addresses it

"""
            
            task_note = ""
            if validation_feedback:
                task_note = "IMPORTANT: Address the validation feedback above to ensure the generated code passes validation."
            
            user_prompt = f"""You are Agent 3A: Dataflow Activity Decision Agent.
Your task is to decide which transformations are needed for each dimension and fact table.

╔═══════════════════════════════════════════════════════════════════════════════╗
║ CRITICAL: DOMAIN-INDEPENDENT DECISION MAKING                                  ║
╚═══════════════════════════════════════════════════════════════════════════════╝

⚠️ IMPORTANT: Make decisions based ONLY on column data types and Agent 1/Agent 2 analysis, 
NOT on domain-specific knowledge. The same patterns work for Sales, Healthcare, HR, Finance, 
Manufacturing, or ANY other domain. Table names (DimProduct, DimPatient, DimEmployee, etc.) 
don't matter - only the column structure and data types matter.

╔═══════════════════════════════════════════════════════════════════════════════╗
║ DECISION LOGIC FOR DATAFLOW ACTIVITIES (WORKS FOR ANY DOMAIN)                ║
╚═══════════════════════════════════════════════════════════════════════════════╝

FOR EACH Dimension Table (regardless of domain):
  1. SELECT - Always required (filter columns for this dimension)
  2. AGGREGATE - Required if:
     - Primary key column exists AND
     - Source CSV has duplicate rows
     Decision: Use groupBy(PK_COLUMN) with first() for all other columns
  3. CAST - Required if:
     - Any column needs type conversion from string (based on Agent 2's adf_type)
  4. DERIVE - Required if:
     - Date format conversion needed (string date → date type)
     Example: Any date column (string "5/7/2003" → date using toDate())

FOR Fact Table (regardless of domain):
  1. SELECT - Always required
  2. DERIVE - Required if date conversion needed (do BEFORE cast)
  3. CAST - Required if numeric columns need type conversion (based on Agent 2's adf_type)

╔═══════════════════════════════════════════════════════════════════════════════╗
║ CRITICAL COLUMN REQUIREMENTS                                                  ║
╚═══════════════════════════════════════════════════════════════════════════════╝

1. For EACH dimension table, include ALL columns from Agent 1's dimension definition
2. For fact table, include ALL columns from Agent 1's fact_columns list
3. Use exact column names from Agent 2's datatype_mapping.json structure
4. Map CSV columns to table columns using exact name matching
5. DO NOT omit any columns - every column in Agent 1's definitions MUST be included

INPUTS:
═══════════════════════════════════════════════════════════════════════════════
Agent 1 Analysis: {json.dumps(csv_analysis, indent=2)}

Agent 2 Analysis: {json.dumps(datatype_analysis, indent=2)}

Target Tables: {json.dumps({k: list(v.keys()) for k, v in destination_tables.items()}, indent=2)}

CSV Data: {len(csv_columns)} columns from CSV

Dimensions: {json.dumps(dimensions, indent=2)}
{validation_section}TASK:
═══════════════════════════════════════════════════════════════════════════════
Analyze each dimension and fact table, then output a JSON decision object.
{task_note}

OUTPUT FORMAT (JSON) - MUST INCLUDE column_mappings:
{{
  "DimProduct": {{
    "activities": ["select", "aggregate", "cast"],
    "aggregate_key": "PRODUCTCODE",
    "column_mappings": {{
      "PRODUCTCODE": "PRODUCTCODE",
      "PRODUCTLINE": "PRODUCTLINE",
      "MSRP": "MSRP"
    }},
    "cast_columns": {{"MSRP": "decimal(10,2)"}},
    "derive_columns": {{}}
  }},
  "DimCustomer": {{
    "activities": ["select", "aggregate"],
    "aggregate_key": "CUSTOMERNAME",
    "column_mappings": {{
      "CUSTOMERNAME": "CUSTOMERNAME",
      "PHONE": "PHONE",
      "ADDRESSLINE1": "ADDRESSLINE1",
      "CITY": "CITY",
      "STATE": "STATE"
    }},
    "cast_columns": {{}},
    "derive_columns": {{}}
  }},
  "DimTime": {{
    "activities": ["select", "aggregate", "derive", "cast"],
    "aggregate_key": "ORDERDATE",
    "column_mappings": {{
      "ORDERDATE": "ORDERDATE",
      "QTR_ID": "QTR_ID",
      "MONTH_ID": "MONTH_ID",
      "YEAR_ID": "YEAR_ID"
    }},
    "cast_columns": {{"QTR_ID": "integer", "MONTH_ID": "integer", "YEAR_ID": "integer"}},
    "derive_columns": {{"ORDERDATE": "toDate(ORDERDATE, 'M/d/yyyy')"}}
  }},
  "FactSales": {{
    "activities": ["select", "derive", "cast"],
    "aggregate_key": null,
    "column_mappings": {{
      "ORDERNUMBER": "ORDERNUMBER",
      "QUANTITYORDERED": "QUANTITYORDERED",
      "PRICEEACH": "PRICEEACH",
      "SALES": "SALES",
      "ORDERDATE": "ORDERDATE"
    }},
    "cast_columns": {{
      "ORDERNUMBER": "integer",
      "QUANTITYORDERED": "integer",
      "PRICEEACH": "decimal(10,2)",
      "SALES": "decimal(15,2)"
    }},
    "derive_columns": {{"ORDERDATE": "toDate(ORDERDATE, 'M/d/yyyy')"}}
  }}
}}

╔═══════════════════════════════════════════════════════════════════════════════╗
║ CRITICAL: ADF DATA FLOW TYPE REQUIREMENTS FOR CAST OPERATIONS                ║
╚═══════════════════════════════════════════════════════════════════════════════╝

ADF Data Flow DSL ONLY supports these types in cast operations:
- string (for all text: VARCHAR, NVARCHAR, CHAR, NCHAR, TEXT, NTEXT)
- integer (for INT, SMALLINT, TINYINT)
- long (for BIGINT)
- double (for FLOAT, REAL)
- decimal(18,2) (for DECIMAL, NUMERIC, MONEY - format: decimal(precision,scale))
- boolean (for BIT)
- timestamp (for DATETIME, DATETIME2, SMALLDATETIME)
- date (for DATE only)
- byte (for BINARY, VARBINARY)
- binary (for IMAGE, VARBINARY(MAX))

CRITICAL: In "cast_columns", you MUST use ADF types, NOT SQL types:
- DO NOT use: nvarchar, varchar, datetime2, etc.
- USE: string, integer, long, double, decimal(18,2), boolean, timestamp, date
- Get ADF types from Agent 2's "adf_type" field (NOT sql_type)

CRITICAL INSTRUCTIONS (DOMAIN-INDEPENDENT):
1. For EACH dimension/fact table, decide which activities are needed based ONLY on:
   - Column data types from Agent 2 (adf_type field)
   - Column structure from Agent 1
   - NOT on domain knowledge or table names
2. "activities" array MUST follow this order if present: ["select", "aggregate", "derive", "cast"]
3. "aggregate_key" is the primary key column for groupBy (NULL for fact tables)
   - Find primary key from Agent 1's dimension definition (primary_key field)
4. "column_mappings" MUST include ALL columns from Agent 1's dimension/fact definitions
   - Map CSV column names (keys) to table column names (values)
   - Include EVERY column listed in Agent 1's dimension columns or fact_columns
   - Use exact name matching (case-sensitive)
   - Works the same for Sales, Healthcare, HR, Finance, or any domain
   - NOTE: Column names with hyphens (e.g., "columns-20", "columns-25") will need special escaping in dataflow scripts (see Agent 3B instructions)
5. "cast_columns" maps column names to ADF types (e.g., {{"MSRP": "decimal(18,2)", "Age": "integer", "Amount": "decimal(10,2)"}})
   - MUST use Agent 2's "adf_type" field (NOT sql_type)
   - Valid ADF types: string, integer, long, double, decimal(18,2), boolean, timestamp, date
   - For decimal, use format: decimal(18,2) or decimal(10,2) with precision from Agent 2
   - Apply same logic regardless of domain
6. "derive_columns" maps column names to derive expressions (e.g., {{"ORDERDATE": "toDate(ORDERDATE, 'M/d/yyyy')"}})
   - Only add if date format conversion is needed (check Agent 2's adf_type for date/timestamp columns)
   - Use appropriate date format pattern based on CSV data format
7. Use Agent 2's datatype analysis "adf_type" field to determine CAST requirements (domain-independent)
8. Use Agent 1's column analysis to determine AGGREGATE requirements (domain-independent)
9. VALIDATION: Count columns in column_mappings - must match Agent 1's column count exactly
10. DOMAIN INDEPENDENCE: The decision logic is the same whether you're processing Sales, Healthcare, HR, Finance, or any other domain. Focus on data types and structure, not domain semantics.

OUTPUT ONLY THE JSON OBJECT, nothing else."""
            
            # Try with JSON mode first
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are an expert in Azure Data Factory dataflow transformations. You analyze schemas and decide which transformations (select, aggregate, derive, cast) are needed for each table. Output ONLY valid JSON."},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.2,
                    max_tokens=16000,
                    response_format={"type": "json_object"}
                )
            except Exception as e:
                # Fallback to regular response if JSON mode not supported
                print(f"JSON mode not supported, trying without: {e}")
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are an expert in Azure Data Factory dataflow transformations. You analyze schemas and decide which transformations (select, aggregate, derive, cast) are needed for each table. Output ONLY valid JSON."},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.2,
                    max_tokens=16000
                )
            
            generated_prompt = response.choices[0].message.content
            # Parse and validate JSON
            try:
                decision_json = json.loads(generated_prompt)
                return decision_json
            except json.JSONDecodeError:
                # Try to extract JSON from markdown or text
                json_match = re.search(r'\{.*\}', generated_prompt, re.DOTALL)
                if json_match:
                    try:
                        decision_json = json.loads(json_match.group())
                        return decision_json
                    except json.JSONDecodeError:
                        pass
                print("Warning: Agent 3A output is not valid JSON, returning None")
                return None
                
        except Exception as e:
            print(f"Error in Agent 3A prompt generation: {type(e).__name__}: {e}")
            traceback.print_exc()
            return None
    
    def generate_python_sdk_code_from_prompt(self, agent3a_decision, csv_analysis=None, 
                                            datatype_analysis=None, agent2_mapping=None,
                                            csv_filename=None, blob_container='applicationdata', 
                                            blob_folder='source', file_name=None, validation_feedback=None,
                                            stream_container=None):
        """
        Agent 3B: Generate complete Python SDK code based on Agent 3A's decision JSON.
        References the provided sample code for consistent structure.
        
        Args:
            agent3a_decision: Decision JSON from Agent 3A with activities and transformations
            csv_analysis: Agent 1 output with fact_columns, dimensions, and column mappings
            datatype_analysis: Agent 2 output with SQL type recommendations
            agent2_mapping: Agent 2's datatype_mapping.json structure with exact column lists
            csv_filename: Full CSV file path from frontend (e.g., 'source/Sunrise_Medical_Center.csv')
            blob_container: Blob container name
            blob_folder: Blob folder path (extracted from csv_filename)
            file_name: CSV filename only (extracted from csv_filename)
            validation_feedback: Optional feedback from Agent 3C validation to fix code issues
            stream_container: Optional Streamlit container for displaying streaming response
        """
        try:
            if self.client is None:
                raise ValueError("OpenAI client is not available")
            
            if not isinstance(agent3a_decision, dict):
                raise ValueError("Agent 3A output must be a dictionary")
            
            # Read the sample code file
            sample_code_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'templates', 'sample_code.py')
            sample_code = ""
            
            if os.path.exists(sample_code_path):
                with open(sample_code_path, 'r', encoding='utf-8') as f:
                    sample_code = f.read()
            
            # If no sample code file, use inline sample from requirements
            if not sample_code:
                sample_code = """# Sample code structure reference
# This shows the expected structure for ADF pipeline generation"""
            
            # Build Agent 1/2 context strings
            agent1_context = ""
            if csv_analysis:
                agent1_context = f"""
AGENT 1 COLUMN MAPPINGS (MANDATORY - USE ALL COLUMNS):
═══════════════════════════════════════════════════════════════════════════════
{json.dumps(csv_analysis, indent=2)}

CRITICAL: Use EXACT column names from Agent 1's dimension definitions and fact_columns list.
"""
            
            agent2_context = ""
            if datatype_analysis:
                agent2_context = f"""
AGENT 2 DATATYPE ANALYSIS (MANDATORY - USE FOR CASTING):
═══════════════════════════════════════════════════════════════════════════════
{json.dumps(datatype_analysis, indent=2)}

CRITICAL: Use Agent 2's SQL type recommendations for cast transformations.
"""
            
            agent2_mapping_context = ""
            if agent2_mapping:
                agent2_mapping_context = f"""
AGENT 2 DATATYPE MAPPING (MANDATORY - EXACT COLUMN STRUCTURE):
═══════════════════════════════════════════════════════════════════════════════
{json.dumps(agent2_mapping, indent=2)}

CRITICAL: This is the EXACT structure from agent2_datatype_mapping.json.
- Use EXACT column names from fact_table.fact_columns
- Use EXACT column names from dimensions[DimName].columns
- Include ALL columns listed - DO NOT omit any
- Column counts must match exactly
"""
            
            csv_file_context = ""
            if csv_filename and file_name:
                csv_file_context = f"""
CSV FILE LOCATION FROM FRONTEND UI (MANDATORY - USE EXACT VALUES):
═══════════════════════════════════════════════════════════════════════════════
- Full Path: {csv_filename}
- Container: {blob_container}
- Folder Path: {blob_folder}
- File Name: {file_name}

CRITICAL INSTRUCTIONS FOR CSV DATASET:
1. In create_source_csv_dataset() method, use EXACT values:
   - container_name='{blob_container}'
   - folder_path='{blob_folder}'
   - file_name='{file_name}'  # ONLY filename, NOT folder path
2. DO NOT hardcode 'healthcare_data_sample.csv' - use the actual filename: '{file_name}'
3. DO NOT include folder path in file_name parameter
4. The folder_path and file_name are already separated correctly above
5. Use these EXACT values in the generated code - they come from the frontend UI selection
"""
            
            # Build validation feedback section separately to avoid nested f-string issues
            validation_feedback_section = ""
            if validation_feedback:
                validation_feedback_section = f"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║ VALIDATION FEEDBACK FROM AGENT 3C (MUST FIX)                                 ║
╚═══════════════════════════════════════════════════════════════════════════════╝

The previous code generation had the following issues that MUST be fixed:

{validation_feedback}

CRITICAL: You MUST fix ALL issues listed above in your generated code.
- If columns were missing, ensure ALL columns are included in dataflow scripts
- If transformations were missing, ensure they are added in the correct order
- If methods were missing, ensure all required methods are implemented
- If code structure was wrong, ensure it matches the sample code structure
- Review each issue carefully and ensure your code addresses it

"""
            
            task_note_3b = ""
            if validation_feedback:
                task_note_3b = "IMPORTANT: Fix all issues from the validation feedback above to ensure the code passes validation."
            
            user_prompt = f"""You are generating Azure Data Factory Python SDK code.

╔═══════════════════════════════════════════════════════════════════════════════╗
║ CRITICAL: NO JOINS OR UNION OPERATIONS                                        ║
║ DO NOT add any joins, unions, or merge operations in dataflow scripts         ║
║ The sample code shows simple source → select → aggregate → sink pattern only  ║
╚═══════════════════════════════════════════════════════════════════════════════╝

REFERENCE SAMPLE CODE STRUCTURE:
═══════════════════════════════════════════════════════════════════════════════
{sample_code}

╔═══════════════════════════════════════════════════════════════════════════════╗
║ SUCCESSFULLY EXECUTED CODE PATTERN (genrated_code.py)                          ║
╚═══════════════════════════════════════════════════════════════════════════════╝

This code successfully executed for Sales domain. Follow these key patterns:

KEY PATTERNS FROM SUCCESSFUL CODE:
1. Source Output: Dimension source combines ALL columns from ALL dimensions
   - PREFERRED: All columns defined as 'string' type initially (like sample_code.py)
   - ACCEPTABLE: Some columns pre-typed if they work correctly (as in genrated_code.py)
   - Example: CUSTOMERNAME as string, PHONE as string, PRODUCTCODE as string, MSRP as string (preferred)
   - Note: genrated_code.py had MSRP as decimal(10,2) in source, but all-strings pattern is preferred

2. Transformation Order: source → select → aggregate → cast → sink
   - Select includes ALL columns for each dimension
   - Aggregate uses groupBy(primary_key) with first() for other columns
   - Cast converts types using ADF types (integer, decimal(10,2), etc.)

3. Transformations Array: Only contains Select*, Aggregate*, Cast*, Derive* names
   - NEVER includes Load* names (those are sinks)
   - Example: [Transformation(name='SelectDimCustomer'), Transformation(name='AggregateDimCustomer')]

4. Sinks Array: Contains Load* names only
   - Example: [DataFlowSink(name='LoadDimProduct'), DataFlowSink(name='LoadDimCustomer')]

5. Fact Dataflow: Similar pattern with derive transformation when needed
   - Source: All fact columns (preferably as string, but pre-typed acceptable if working)
   - Derive: Only if date conversion needed (derive(ORDERDATE = toDate(ORDERDATE, 'M/d/yyyy')))
   - Cast: Converts numeric/date types using ADF types

⚠️ IMPORTANT: These patterns work for ANY domain. Apply the same structure to Healthcare, HR, Finance, etc.
⚠️ RECOMMENDATION: Use all-strings pattern in source output (like sample_code.py) for consistency across domains.

{agent1_context}
{agent2_context}
{agent2_mapping_context}
{csv_file_context}
AGENT 3A DECISION LOGIC (which transformations to use):
═══════════════════════════════════════════════════════════════════════════════
{json.dumps(agent3a_decision, indent=2)}
{validation_feedback_section}
╔═══════════════════════════════════════════════════════════════════════════════╗
║ CRITICAL: COLUMN NAME ESCAPING FOR SPECIAL CHARACTERS                         ║
╚═══════════════════════════════════════════════════════════════════════════════╝

⚠️ CRITICAL RULE: Column names with hyphens or special characters MUST be escaped in dataflow scripts.
In Azure Data Factory dataflow scripts, column names containing hyphens (like "columns-20", "columns-25") 
or other special characters MUST be enclosed in double curly braces {{}}.

CORRECT PATTERN:
- Column name: "columns-25" → Use: {{columns-25}} in dataflow script
- Column name: "column-name" → Use: {{column-name}} in dataflow script
- Column name: "normal_column" → Use: normal_column (no escaping needed)

EXAMPLES IN DATAFLOW SCRIPT:
source(output(
      {{columns-20}} as string,
      {{columns-25}} as string,
      normal_column as string
),
...) ~> SourceCSV

SourceCSV select(mapColumn(
      {{columns-20}},
      {{columns-25}},
      normal_column
)) ~> SelectTable

SelectTable aggregate(groupBy({{columns-20}}),
 {{columns-25}} = first({{columns-25}}),
 normal_column = first(normal_column)
) ~> AggregateTable

⚠️ IMPORTANT: Check ALL column names from Agent 1/Agent 2 - if any contain hyphens, escape them with {{}}.

╔═══════════════════════════════════════════════════════════════════════════════╗
║ CRITICAL: SOURCE OUTPUT TYPE REQUIREMENT (DOMAIN-INDEPENDENT)                 ║
╚═══════════════════════════════════════════════════════════════════════════════╝

⚠️ CRITICAL RULE: Source output in dataflow scripts MUST have ALL columns as 'string' type initially.
CSV files are read as text, so source output should always be string initially. Type conversion 
happens in cast() transformations later, NOT in source output.

CORRECT PATTERN (from sample_code.py):
source(output(
      COLUMN1 as string,
      COLUMN2 as string,
      COLUMN3 as string,
      NUMERIC_COLUMN as string,  # Even numeric columns start as string
      DATE_COLUMN as string       # Even date columns start as string
),
allowSchemaDrift: true,
validateSchema: false,
ignoreNoFilesFound: false) ~> SourceCSV

WRONG PATTERN (DO NOT DO THIS):
source(output(
      COLUMN1 as string,
      NUMERIC_COLUMN as integer,  # ❌ WRONG - don't pre-type in source
      DATE_COLUMN as date          # ❌ WRONG - don't pre-type in source
),
...

╔═══════════════════════════════════════════════════════════════════════════════╗
║ HOW TO BUILD SOURCE OUTPUT (STEP-BY-STEP)                                     ║
╚═══════════════════════════════════════════════════════════════════════════════╝

FOR DIMENSION DATAFLOW SOURCE:
Step 1: Collect ALL columns from ALL dimension tables
   - Loop through Agent 1's dimensions dictionary
   - For each dimension, get ALL columns from dimensions[DimName].columns
   - Combine all columns into a single list
   - Remove duplicates if same column appears in multiple dimensions (keep only one)
   
Step 2: Use exact column names from Agent 1/Agent 2
   - Use the exact column names as they appear in Agent 1's dimension definitions
   - Use exact column names from Agent 2's datatype_mapping.json
   - DO NOT modify column names
   
Step 3: Define ALL columns as 'string' type in source output
   - Every column in source output must be: ColumnName as string
   - This works for ANY domain (Sales, Healthcare, HR, Finance, etc.)

Example Dimension Source Output:
source(output(
      Dim1Col1 as string,
      Dim1Col2 as string,
      Dim2Col1 as string,
      Dim2Col2 as string,
      SharedCol as string  # If column appears in multiple dimensions, include once
),
allowSchemaDrift: true,
validateSchema: false,
ignoreNoFilesFound: false) ~> SourceCSV

FOR FACT DATAFLOW SOURCE:
Step 1: Collect ALL columns from fact table
   - Get ALL columns from Agent 1's fact_columns list
   - Use exact column names from Agent 2's datatype_mapping.json (fact_table.fact_columns)
   
Step 2: Define ALL columns as 'string' type in source output
   - Every column must be: ColumnName as string
   - This works for ANY domain

Example Fact Source Output:
source(output(
      FactCol1 as string,
      FactCol2 as string,
      NumericCol as string,  # Even numeric - starts as string
      DateCol as string      # Even date - starts as string
),
allowSchemaDrift: true,
validateSchema: false,
ignoreNoFilesFound: false) ~> SourceCSV

╔═══════════════════════════════════════════════════════════════════════════════╗
║ CRITICAL COLUMN REQUIREMENTS (DOMAIN-INDEPENDENT)                             ║
╚═══════════════════════════════════════════════════════════════════════════════╝

1. Dimension Dataflow source MUST include ALL columns needed for ALL dimensions
   - Combine all columns from ALL dimension tables (from Agent 1's dimensions)
   - Include every column listed in Agent 1's dimension definitions
   - Use exact column names from Agent 2's datatype_mapping.json
   - ALL columns must be defined as 'string' type in source output
   - Remove duplicates if same column appears in multiple dimensions

2. Each dimension's select MUST include ALL columns from Agent 1's dimension definition
   - Include EVERY column listed in dimensions[DimName].columns
   - Use exact column names (case-sensitive)
   - Works the same for ANY domain (Sales, Healthcare, HR, Finance, etc.)
   - Example: If a dimension has 10 columns, select MUST include all 10

3. Fact dataflow source MUST include ALL columns from Agent 1's fact_columns
   - Include every column listed in fact_table.fact_columns from Agent 2 mapping
   - Use exact column names from Agent 2's datatype_mapping.json
   - ALL columns must be defined as 'string' type in source output
   - Works the same for ANY domain

4. Use exact column names from Agent 2's datatype_mapping.json
   - DO NOT change column names
   - DO NOT omit any columns
   - DO NOT add columns not in Agent 1/Agent 2 outputs
   - Column names are domain-independent - same rules apply to all domains
   - CRITICAL: If column names contain hyphens (e.g., "columns-20", "columns-25"), escape them with {{}} in dataflow scripts
   - Example: "columns-25" → use {{columns-25}} in all dataflow script operations (select, aggregate, derive, cast)

5. Verify column counts match Agent 1/Agent 2 outputs exactly

VALIDATION CHECKLIST (MANDATORY - DOMAIN-INDEPENDENT):
- [ ] Dimension source has ALL columns from ALL dimensions combined (works for any domain)
- [ ] Each dimension's select has ALL columns from Agent 1's dimension definition (check count)
- [ ] Fact source has ALL columns from fact_table.fact_columns (check count)
- [ ] All columns from agent2_datatype_mapping.json are present in dataflow scripts
- [ ] No columns are missing or omitted
- [ ] Column names match exactly (case-sensitive)
- [ ] Source output has ALL columns as 'string' type (not pre-typed)

TASK:
═══════════════════════════════════════════════════════════════════════════════
Generate COMPLETE Python SDK code that implements the decision logic above.
{task_note_3b}

MANDATORY REQUIREMENTS:
1. Follow sample code structure EXACTLY - no deviations
2. Generate TWO dataflows: one for ALL dimensions, one for fact table
3. Dimensions load FIRST, then fact table (dependency in pipeline)
4. Build dataflow scripts dynamically based on Agent 3A's "activities" arrays
5. Activity flow pattern: source → select → aggregate (if needed) → derive (ONLY if derive_columns not empty, BEFORE cast) → cast (if needed) → sink
   ⚠️ CRITICAL: If derive_columns is empty, SKIP derive transformation - do NOT generate empty derive()
6. NO JOIN operations, NO UNION operations, NO MERGE operations
7. Hardcode ALL Azure credentials and configuration in the code
8. Include all datasets, linked services, and configurations
9. Use proper resource naming and dependency management
10. Generate fully executable code - no placeholders
11. Include ALL columns from Agent 1/Agent 2 in dataflow scripts - NO MISSING COLUMNS

╔═══════════════════════════════════════════════════════════════════════════════╗
║ CRITICAL: ADF DATA FLOW TYPE REQUIREMENTS FOR CAST OPERATIONS                ║
╚═══════════════════════════════════════════════════════════════════════════════╝

ADF Data Flow DSL ONLY supports these types in cast operations:
- string (for all text: VARCHAR, NVARCHAR, CHAR, NCHAR, TEXT, NTEXT)
- integer (for INT, SMALLINT, TINYINT)
- long (for BIGINT)
- double (for FLOAT, REAL)
- decimal(18,2) (for DECIMAL, NUMERIC, MONEY - format: decimal(precision,scale))
- boolean (for BIT)
- timestamp (for DATETIME, DATETIME2, SMALLDATETIME)
- date (for DATE only)
- byte (for BINARY, VARBINARY)
- binary (for IMAGE, VARBINARY(MAX))

CRITICAL CAST RULES:
1. In cast() operations, use ONLY ADF types from Agent 3A's cast_columns
2. Agent 3A's cast_columns already contains ADF types (string, integer, decimal(18,2), etc.)
3. DO NOT use SQL types like: nvarchar, varchar, datetime2, nvarchar(255), etc.
4. Example CORRECT: cast(output(TableID as string, Amount as decimal(18,2), Quantity as integer))
5. Example WRONG: cast(output(TableID as nvarchar, Amount as decimal(18,2)))  ❌
6. For text columns, use: string (not nvarchar, varchar, etc.)
7. For decimal, use format: decimal(18,2) or decimal(10,2) with precision
8. For dates, use: date (for DATE) or timestamp (for DATETIME/DATETIME2)

DATAFLOW SCRIPT GENERATION:
For each dimension/fact table, build the script based on the "activities" array:
- "select": Always first - select mapColumn for ALL dimension/fact columns from Agent 1
  ⚠️ CRITICAL: If column names contain hyphens (e.g., "columns-20", "columns-25"), escape them with {{}}
  ⚠️ Example: select(mapColumn({{columns-20}}, {{columns-25}}, normal_column))
- "aggregate": Use groupBy(aggregate_key) with first() for other columns
  ⚠️ CRITICAL: Escape column names with hyphens in groupBy and first() expressions
  ⚠️ Example: aggregate(groupBy({{columns-20}}), {{columns-25}} = first({{columns-25}}))
- "derive": 
  ⚠️ CRITICAL: ONLY add derive transformation if derive_columns is NOT empty
  ⚠️ If derive_columns is empty {{}}, SKIP the derive transformation entirely - DO NOT generate it
  ⚠️ NEVER generate: derive() ~> DeriveX (empty derive will cause "missing input stream" error in ADF)
  ⚠️ ONLY generate: derive(Column1 = expression1, Column2 = expression2) ~> DeriveX when expressions exist
  ⚠️ CRITICAL: If column names contain hyphens, escape them: derive({{date-column}} = toDate({{date-column}}, 'M/d/yyyy'))
  ⚠️ Example CORRECT: derive(DateColumn = toDate(DateColumn, 'M/d/yyyy')) ~> DeriveFactTable
  ⚠️ Example CORRECT: derive({{date-column}} = toDate({{date-column}}, 'M/d/yyyy')) ~> DeriveFactTable (with hyphen)
  ⚠️ Example WRONG: derive() ~> DeriveFactTable (DO NOT DO THIS - causes deployment failure)
- "cast": Use cast(output(...)) with ADF types from cast_columns (string, integer, decimal(18,2), etc.)
  ⚠️ CRITICAL: If column names contain hyphens, escape them in cast output
  ⚠️ Example: cast(output({{columns-20}} as string, {{columns-25}} as integer))
- "sink": Always last - sink to table

EXAMPLE for ANY Dimension Table (works for Sales, Healthcare, HR, Finance, etc.):
Generic pattern - replace DimTable1 with actual dimension name from Agent 1:

script = \"\"\"
SourceCSV select(mapColumn(
      Column1,
      Column2,
      Column3,
      ... (ALL columns from Agent 1's dimension definition)
)) ~> SelectDimTable1
SelectDimTable1 aggregate(groupBy(PrimaryKeyColumn),
 Column1 = first(Column1),
 Column2 = first(Column2),
 ... (all other columns with first())
) ~> AggregateDimTable1
AggregateDimTable1 cast(output(
      NumericColumn as integer,  # or decimal(18,2), etc. based on Agent 2's adf_type
      DateColumn as date         # if date conversion needed
), errors: true) ~> CastDimTable1

NOTE: In cast operations, use ADF types: string, integer, long, double, decimal(18,2), boolean, timestamp, date
DO NOT use SQL types like: nvarchar, varchar, datetime2, etc.
CastDimTable1 sink(...) ~> LoadDimTable1
\"\"\"

⚠️ IMPORTANT: This example pattern works for ANY domain. Replace:
- DimTable1 with actual dimension name (DimProduct, DimCustomer, DimPatient, DimEmployee, etc.)
- Column names with actual column names from Agent 1/Agent 2
- PrimaryKeyColumn with actual primary key from Agent 1

REQUIRED CLASS STRUCTURE:
- generate_resource_names(): Return dict with Neccessory resources names as per agent3a_decision 
- get_credential(): Return ClientSecretCredential only from def main() function
- create_sql_linked_service(): Create Azure SQL linked service
- create_blob_storage_linked_service(): Create blob storage linked service
- create_source_csv_dataset(): Create source CSV dataset
- create_sql_datasets(): Create ALL dimension and fact datasets
- create_dimension_dataflow(): Create dataflow for ALL dimensions with ALL columns from Agent 1/Agent 2
- create_fact_dataflow(): Create dataflow for fact table with ALL columns from Agent 1/Agent 2
- create_pipeline(): Create pipeline with proper dependencies
- deploy_complete_solution(): Orchestrate full deployment
- run_pipeline(): Execute pipeline
- monitor_pipeline(): Monitor execution

CRITICAL: TRANSFORMATIONS vs SINKS DISTINCTION:
═══════════════════════════════════════════════════════════════════════════════
In Azure Data Factory data flows, there is a CRITICAL distinction:

TRANSFORMATIONS (operations that modify data):
- Select* (e.g., SelectDimProduct, SelectFactSales)
- Aggregate* (e.g., AggregateDimProduct)
- Cast* (e.g., CastDimProduct, CastFactSales)
- Derive* (e.g., DeriveDimTime)
- These go in: transformations=[Transformation(name='SelectDimProduct'), ...]

SINKS (final destinations where data is written):
- Load* (e.g., LoadDimProduct, LoadFactSales, LoadDimBusinessGroup, LoadFactEmployeeMetrics)
- These go in: sinks=[DataFlowSink(name='LoadDimProduct'), ...]

⚠️ CRITICAL RULES:
1. NEVER include Load* names in the transformations array
2. Load* names should ONLY appear in the sinks array
3. If you see "~> LoadSomething" in the script, it's a SINK, not a transformation
4. Transformations array should ONLY contain: Select*, Aggregate*, Cast*, Derive* names
5. When building transformations list, extract names from script but EXCLUDE any name starting with "Load"

CORRECT EXAMPLE:
transformations=[
    Transformation(name='SelectDimProduct'),
    Transformation(name='AggregateDimProduct'),
    Transformation(name='CastDimProduct')
],
sinks=[
    DataFlowSink(name='LoadDimProduct')  # Load* is a sink, NOT a transformation
]

WRONG EXAMPLE (DO NOT DO THIS):
transformations=[
    Transformation(name='SelectDimProduct'),
    Transformation(name='LoadDimProduct')  # ❌ WRONG - Load* is a sink!
]

Generate ONLY the Python code, starting with the class definition and including all methods."""
            
            system_prompt = self.COMPLETE_AGENT_3_SYSTEM_PROMPT + "\n\n" + """You generate complete, working Python SDK code for Azure Data Factory. 
CRITICAL RULES:
1. Follow the sample code structure EXACTLY
2. NEVER add joins, unions, or merge operations in dataflow scripts
3. Use simple pattern: source → select → aggregate → derive (ONLY if derive_columns not empty) → cast → sink (based on Agent 3A decisions)
4. Hardcode all credentials and configuration
5. Generate fully executable code with no placeholders
6. Build transformations dynamically based on Agent 3A's decision JSON
7. CRITICAL: Source output in dataflow scripts MUST have ALL columns as 'string' type initially
   - CSV files are read as text, so source output should always be string initially
   - Type conversion happens in cast() transformations later, NOT in source output
   - This pattern works for ANY domain (Sales, Healthcare, HR, Finance, etc.)
8. CRITICAL: In cast() operations, use ONLY ADF types: string, integer, long, double, decimal(18,2), boolean, timestamp, date
9. CRITICAL: NEVER use SQL types in cast operations: nvarchar, varchar, datetime2, etc. - these will cause deployment failures
10. Agent 3A's cast_columns already contains ADF types - use them directly in cast() operations
11. CRITICAL: NEVER generate empty derive() transformations - if derive_columns is empty, SKIP derive transformation entirely
12. CRITICAL: Empty derive() like "derive() ~>" causes "missing input stream" error - always check if derive_columns has expressions before adding derive
13. CRITICAL: NEVER include Load* names in transformations array - Load* names are sinks, not transformations
14. CRITICAL: Transformations array should only contain: Select*, Aggregate*, Cast*, Derive* names
15. CRITICAL: Load* names (like LoadDimProduct, LoadFactSales, LoadDimBusinessGroup) belong ONLY in sinks array
16. CRITICAL: When extracting transformation names from script, EXCLUDE any name starting with "Load"
17. CRITICAL: Generate domain-independent code - same patterns work for Sales, Healthcare, HR, Finance, Manufacturing, or ANY domain
18. CRITICAL: For dimension dataflow source, combine ALL columns from ALL dimensions and define ALL as 'string' type
19. CRITICAL: For fact dataflow source, include ALL columns from fact_columns and define ALL as 'string' type
20. CRITICAL: Column names with hyphens (e.g., "columns-20", "columns-25") MUST be escaped with {{}} in dataflow scripts
    - In source output: {{columns-20}} as string
    - In select: mapColumn({{columns-20}})
    - In aggregate: groupBy({{columns-20}}), {{columns-25}} = first({{columns-25}})
    - In derive: {{date-column}} = toDate({{date-column}}, 'M/d/yyyy')
    - In cast: cast(output({{columns-20}} as string))
"""
            
            messages = [{"role": "user", "content": user_prompt}]
            
            # Use streaming if stream_container is provided
            if stream_container:
                try:
                    generated_code = self._stream_chat_completion(
                        messages=messages,
                        system_message=system_prompt,
                        temperature=0.1,
                        max_tokens=16000,
                        stream_container=stream_container,
                        show_in_container=True
                    )
                except Exception as stream_error:
                    print(f"Streaming failed, falling back to non-streaming: {stream_error}")
                    # Fallback to non-streaming
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ],
                        temperature=0.1,
                        max_tokens=16000
                    )
                    generated_code = response.choices[0].message.content
            else:
                # Non-streaming mode
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.1,
                    max_tokens=16000
                )
                generated_code = response.choices[0].message.content
            
            # Extract code from markdown if present
            if '```' in generated_code:
                code_pattern = r'```(?:python)?\s*\n(.*?)\n```'
                matches = re.findall(code_pattern, generated_code, re.DOTALL)
                if matches:
                    generated_code = matches[0].strip()
            
            if not generated_code or len(generated_code.strip()) == 0:
                raise ValueError("Generated code is empty")
            
            return generated_code
            
        except Exception as e:
            error_msg = f"Error in Agent 3B code generation: {type(e).__name__}: {str(e)}"
            print(error_msg)
            traceback.print_exc()
            raise Exception(error_msg) from e
    
    # ==================== AGENT 3C: CODE VALIDATION ====================
    
    def validate_generated_code(self, generated_code, agent3a_decision, csv_analysis=None, 
                                datatype_analysis=None, agent2_mapping=None, sample_code=None):
        """
        Agent 3C: Validates the generated ADF code against requirements from Agents 3A & 3B prompts.
        
        Args:
            generated_code: Python SDK code generated by Agent 3B
            agent3a_decision: Decision JSON from Agent 3A
            csv_analysis: Agent 1 output with fact/dimension structure
            datatype_analysis: Agent 2 output with SQL type recommendations
            agent2_mapping: Agent 2's datatype_mapping.json structure
            sample_code: Reference sample code structure
            
        Returns:
            dict with:
                - is_valid: boolean indicating if code passes validation
                - issues: list of specific issues found
                - feedback: formatted feedback string for Agents 3A & 3B
                - validation_details: detailed breakdown of checks
        """
        try:
            if self.client is None:
                # Fallback validation without AI
                return {
                    "is_valid": True,
                    "issues": [],
                    "feedback": "Validation skipped - OpenAI client not available",
                    "validation_details": {}
                }
            
            if not generated_code or len(generated_code.strip()) == 0:
                return {
                    "is_valid": False,
                    "issues": ["Generated code is empty"],
                    "feedback": "The generated code is empty. Please regenerate the code.",
                    "validation_details": {}
                }
            
            # ==================== REGEX-BASED PRE-CHECKS (Domain-Independent) ====================
            pre_check_issues = []
            pre_check_details = {
                "method_signature": {"found": False, "has_sql_config": False, "has_blob_config": False},
                "syntax_errors": False,
                "sql_types_in_cast": []
            }
            
            # Pre-check 1: Method signature validation (DOMAIN-INDEPENDENT)
            import re
            deploy_method_pattern = r'def\s+deploy_complete_solution\s*\([^)]*\)'
            deploy_match = re.search(deploy_method_pattern, generated_code, re.IGNORECASE | re.MULTILINE)
            
            if deploy_match:
                method_signature = deploy_match.group(0)
                pre_check_details["method_signature"]["found"] = True
                
                # Check for sql_config parameter (case-insensitive, allows variations)
                if re.search(r'\bsql_config\b', method_signature, re.IGNORECASE):
                    pre_check_details["method_signature"]["has_sql_config"] = True
                
                # Check for blob_config parameter (case-insensitive, allows variations)
                if re.search(r'\bblob_config\b', method_signature, re.IGNORECASE):
                    pre_check_details["method_signature"]["has_blob_config"] = True
                
                # Flag if parameters are missing
                if not pre_check_details["method_signature"]["has_sql_config"]:
                    pre_check_issues.append("The 'deploy_complete_solution' method is missing required parameter: 'sql_config'")
                if not pre_check_details["method_signature"]["has_blob_config"]:
                    pre_check_issues.append("The 'deploy_complete_solution' method is missing required parameter: 'blob_config'")
            else:
                pre_check_issues.append("The 'deploy_complete_solution' method definition was not found in the generated code")
            
            # Pre-check 2: SQL types in cast operations (DOMAIN-INDEPENDENT)
            # Look for cast operations with SQL-specific types
            sql_type_patterns = [
                r'cast\s*\([^)]*as\s+(nvarchar|varchar|datetime2|datetime|char|nchar|text|ntext)',
                r'cast\s*\([^)]*as\s+(nvarchar|varchar|datetime2|datetime|char|nchar|text|ntext)\s*\(',
            ]
            
            for pattern in sql_type_patterns:
                matches = re.finditer(pattern, generated_code, re.IGNORECASE | re.MULTILINE)
                for match in matches:
                    sql_type = match.group(1)
                    if sql_type not in pre_check_details["sql_types_in_cast"]:
                        pre_check_details["sql_types_in_cast"].append(sql_type)
                        pre_check_issues.append(f"SQL type '{sql_type}' found in cast operation - ADF only supports: string, integer, long, double, decimal, boolean, timestamp, date")
            
            # Pre-check 3: Basic syntax validation (DOMAIN-INDEPENDENT)
            try:
                compile(generated_code, '<string>', 'exec')
            except SyntaxError as e:
                pre_check_details["syntax_errors"] = True
                pre_check_issues.append(f"Python syntax error: {str(e)}")
            
            # Pre-check 4: Empty derive() validation (DOMAIN-INDEPENDENT)
            empty_derive_pattern = r'derive\s*\(\s*\)\s*~>'
            empty_derive_matches = re.finditer(empty_derive_pattern, generated_code, re.IGNORECASE | re.MULTILINE)
            for match in empty_derive_matches:
                pre_check_issues.append("Empty derive() transformation found - derive() must have expressions or be removed. This causes 'missing input stream' error in ADF.")
                break  # Only flag once
            
            # Pre-check 5: Load* names in transformations validation (DOMAIN-INDEPENDENT)
            load_in_transformations_pattern = r"Transformation\(name=['\"]Load\w+['\"]\)"
            load_matches = re.finditer(load_in_transformations_pattern, generated_code, re.IGNORECASE | re.MULTILINE)
            for match in load_matches:
                pre_check_issues.append("Load* name found in transformations array - Load* names are sinks, not transformations. This causes 'missing input stream' error in ADF.")
                break  # Only flag once
            
            # If pre-checks found critical issues, return early (skip AI validation for obvious errors)
            if pre_check_issues:
                return {
                    "is_valid": False,
                    "issues": pre_check_issues,
                    "feedback": "Pre-validation checks failed:\n" + "\n".join(f"  - {issue}" for issue in pre_check_issues),
                    "validation_details": {
                        "pre_check_results": pre_check_details,
                        "code_structure": {
                            "has_all_methods": pre_check_details["method_signature"]["found"],
                            "missing_methods": [] if pre_check_details["method_signature"]["found"] else ["deploy_complete_solution"],
                            "syntax_valid": not pre_check_details["syntax_errors"]
                        },
                        "deployment_blockers": {
                            "sql_types_in_cast": len(pre_check_details["sql_types_in_cast"]) > 0,
                            "sql_types_found": pre_check_details["sql_types_in_cast"],
                            "syntax_errors": pre_check_details["syntax_errors"],
                            "missing_imports": False,
                            "forbidden_operations": False
                        },
                        "method_signatures": {
                            "deploy_complete_solution_valid": (
                                pre_check_details["method_signature"]["found"] and
                                pre_check_details["method_signature"]["has_sql_config"] and
                                pre_check_details["method_signature"]["has_blob_config"]
                            ),
                            "has_sql_config": pre_check_details["method_signature"]["has_sql_config"],
                            "has_blob_config": pre_check_details["method_signature"]["has_blob_config"],
                            "signature_issues": pre_check_issues if not (
                                pre_check_details["method_signature"]["has_sql_config"] and
                                pre_check_details["method_signature"]["has_blob_config"]
                            ) else []
                        }
                    }
                }
            
            # ==================== CONTINUE WITH AI VALIDATION ====================
            # (Only if pre-checks pass - this reduces false positives from AI)
            
            # Build context for validation
            agent1_context = ""
            if csv_analysis:
                agent1_context = f"""
AGENT 1 ANALYSIS (REQUIREMENTS):
═══════════════════════════════════════════════════════════════════════════════
{json.dumps(csv_analysis, indent=2)}

CRITICAL REQUIREMENTS FROM AGENT 1:
- All columns from dimensions must be included
- All columns from fact_columns must be included
- Column names must match exactly (case-sensitive)
- Column counts must match exactly
"""
            
            agent2_context = ""
            if datatype_analysis:
                agent2_context = f"""
AGENT 2 DATATYPE ANALYSIS (REQUIREMENTS):
═══════════════════════════════════════════════════════════════════════════════
{json.dumps(datatype_analysis, indent=2)}

CRITICAL REQUIREMENTS FROM AGENT 2:
- Use exact SQL types from datatype_analysis for cast transformations
- Column names must match datatype_mapping.json structure
"""
            
            agent2_mapping_context = ""
            if agent2_mapping:
                agent2_mapping_context = f"""
AGENT 2 DATATYPE MAPPING (EXACT STRUCTURE):
═══════════════════════════════════════════════════════════════════════════════
{json.dumps(agent2_mapping, indent=2)}

CRITICAL REQUIREMENTS:
- Use EXACT column names from fact_table.fact_columns
- Use EXACT column names from dimensions[DimName].columns
- Include ALL columns listed - DO NOT omit any
- Column counts must match exactly
"""
            
            agent3a_context = ""
            if agent3a_decision:
                agent3a_context = f"""
AGENT 3A DECISION (TRANSFORMATION REQUIREMENTS):
═══════════════════════════════════════════════════════════════════════════════
{json.dumps(agent3a_decision, indent=2)}

CRITICAL REQUIREMENTS FROM AGENT 3A:
- Transformations must match the "activities" arrays
- Column mappings must match column_mappings
- Cast transformations must match cast_columns
- Derive transformations must match derive_columns
- Aggregate keys must match aggregate_key
"""
            
            # Extract deploy_complete_solution method from sample code for reference
            sample_deploy_method = ""
            if sample_code:
                # Extract the deploy_complete_solution method from sample code
                deploy_pattern = r'def\s+deploy_complete_solution\s*\([^)]*\):.*?(?=\n\s{4}def\s|\nclass\s|\Z)'
                deploy_match = re.search(deploy_pattern, sample_code, re.DOTALL)
                if deploy_match:
                    sample_deploy_method = deploy_match.group(0)[:1500]  # Limit to 1500 chars
            
            # Extract sample code comments/docstrings pattern for reference
            sample_comments_example = ""
            if sample_code:
                # Extract a representative section showing acceptable comment style
                # Look for docstrings and comments in sample code
                lines = sample_code.split('\n')[:100]  # First 100 lines
                comment_lines = [line for line in lines if line.strip().startswith('#') or '"""' in line or "'''" in line]
                if comment_lines:
                    sample_comments_example = '\n'.join(comment_lines[:20])  # Show first 20 comment/docstring lines
            
            sample_context = ""
            if sample_code:
                sample_context = f"""
REFERENCE SAMPLE CODE STRUCTURE (FULL CONTEXT):
═══════════════════════════════════════════════════════════════════════════════
{sample_code}

CRITICAL REQUIREMENTS FROM SAMPLE:
- Code structure must match sample code
- Required methods must exist
- Dataflow scripts must follow correct pattern

REFERENCE: CORRECT deploy_complete_solution METHOD SIGNATURE FROM SAMPLE:
═══════════════════════════════════════════════════════════════════════════════
{sample_deploy_method if sample_deploy_method else "Method not found in sample code"}

CRITICAL: The sample code shows the CORRECT signature. Compare the generated code's 
deploy_complete_solution method against this reference. The method MUST have:
- Parameter 1: sql_config (dict)
- Parameter 2: blob_config (dict)
- Signature: def deploy_complete_solution(self, sql_config, blob_config):

REFERENCE: ACCEPTABLE COMMENT STYLE FROM SAMPLE CODE:
═══════════════════════════════════════════════════════════════════════════════
{sample_comments_example if sample_comments_example else "No comments found in sample"}

CRITICAL: The sample code shows ACCEPTABLE comment/docstring style. Comments like:
- Method docstrings explaining parameters and return values: ACCEPTABLE
- Section headers like "# ==================== Linked Services ====================": ACCEPTABLE  
- Print statements with descriptive messages: ACCEPTABLE
- Brief inline comments explaining "why": ACCEPTABLE

ONLY flag comments that are:
- Excessive instructional text (e.g., "# Step 1: Do this, Step 2: Do that" in multiple places)
- TODO/FIXME comments that shouldn't be in production
- Redundant comments that just repeat what the code clearly shows
- Long explanatory blocks that duplicate documentation
"""
            
            validation_prompt = f"""You are Agent 3C: Code Validation Agent.
Your task is to validate the generated Azure Data Factory Python SDK code for DEPLOYMENT-BLOCKING ISSUES ONLY.

╔═══════════════════════════════════════════════════════════════════════════════╗
║ CRITICAL PRINCIPLE: Only flag issues that would cause deployment or runtime   ║
║ failures. Accept code variations that work correctly, even if they differ     ║
║ from expected structure. Verify issues exist in code before flagging them.    ║
╚═══════════════════════════════════════════════════════════════════════════════╝

╔═══════════════════════════════════════════════════════════════════════════════╗
║ VALIDATION CRITERIA (DEPLOYMENT-BLOCKING ISSUES ONLY)                         ║
╚═══════════════════════════════════════════════════════════════════════════════╝

1. CRITICAL DEPLOYMENT BLOCKERS (ALWAYS FLAG):
   ✗ Python syntax errors (would prevent code execution)
   ✗ Missing required Azure SDK imports (would cause ImportError at runtime)
   ✗ SQL types in cast operations (nvarchar, varchar, datetime2, etc.) - WILL cause ADF deployment failure
   ✗ Missing critical methods that are called but not defined (would cause AttributeError)
   ✗ Empty derive() transformations (derive() with no expressions) - WILL cause "missing input stream" error in ADF
   ✗ Invalid ADF dataflow script syntax (malformed script strings)
   ✗ Forbidden operations in dataflow scripts (join, union, merge) if explicitly prohibited
   ✗ Unescaped column names with hyphens in dataflow scripts - column names like "columns-20", "columns-25" MUST be escaped as {{columns-20}}, {{columns-25}}

2. METHOD EXISTENCE VALIDATION (BE LENIENT):
   ✓ Check for required methods case-INSENSITIVELY (create_dimension_dataflow = Create_Dimension_Dataflow)
   ✓ Only flag if method is called but doesn't exist
   ✓ Accept method name variations if they serve the same purpose
   ✓ Don't flag if methods exist but are named slightly differently

3. COLUMN VALIDATION (VERIFY ACTUAL EXISTENCE):
   ✓ BEFORE flagging missing columns, SEARCH the code to verify they don't exist
   ✓ Check source CSV output definition - columns may be defined there
   ✓ Check select transformations - columns may be included
   ✓ Only flag if column is explicitly required but completely absent from code
   ✓ Don't flag based solely on column counts - if columns are present, accept it
   ✓ Column names may vary slightly (case, underscores) - accept reasonable variations

4. TRANSFORMATION VALIDATION (FOCUS ON FUNCTIONALITY):
   ✓ Don't strictly enforce transformation order if the code works
   ✓ Don't flag missing transformations if the dataflow script is functionally correct
   ✓ Accept code variations - if select → aggregate → sink works, don't require cast/derive
   ✓ Only flag if transformation is explicitly required AND causes functionality issues
   ✓ Cast transformations: Only flag if SQL types are used (deployment blocker)
   ✓ Derive transformations: 
     - CRITICAL: Flag empty derive() transformations (derive() with no expressions) - causes "missing input stream" error
     - Only flag if required for data correctness AND missing
     - If derive_columns is empty in Agent 3A decision, derive transformation should be SKIPPED entirely

5. DATAFLOW SCRIPT VALIDATION (SYNTAX AND TYPES ONLY):
   ✓ Verify script syntax is valid (proper chaining with ~>)
   ✓ CRITICAL: Check for SQL types in cast operations (nvarchar, varchar, datetime2, nvarchar(255), etc.)
   ✓ Accept ADF types: string, integer, long, double, decimal(18,2), boolean, timestamp, date
   ✓ Source output validation:
     - PREFERRED: All columns in source output should be 'string' type (like sample_code.py)
     - ACCEPTABLE: Pre-typed columns in source output if they work correctly
     - FLAG: SQL types in source output (nvarchar, varchar, datetime2, etc.) - should use ADF types or string
   ✓ Don't flag join/union/merge if they're not explicitly forbidden
   ✓ Verify sinks exist for dimensions and fact table
   ✓ Don't enforce strict transformation patterns if the script is valid

6. DATAFLOW STRUCTURE VALIDATION (CRITICAL):
   ✗ Load* names in transformations array - Load* names are sinks, not transformations
   ✗ This causes "missing input stream" error in ADF deployment
   ✓ Transformations array should only contain: Select*, Aggregate*, Cast*, Derive*
   ✓ Load* names (LoadDimProduct, LoadFactSales, etc.) should ONLY appear in sinks array
   ✓ When validating, check that no Transformation(name='Load*') exists in code

7. CODE QUALITY (RUNTIME ERRORS ONLY):
   ✓ Flag syntax errors
   ✓ Flag missing imports that would cause ImportError
   ✓ Flag missing parameters that would cause TypeError
   ✓ Don't flag style differences or code organization variations
   ✓ Don't flag placeholders/TODOs unless they cause runtime errors

8. CODE CLEANLINESS VALIDATION (COMPARE AGAINST SAMPLE CODE):
   ═══════════════════════════════════════════════════════════════════════════════
   STEP-BY-STEP VALIDATION PROCESS:
   ═══════════════════════════════════════════════════════════════════════════════
   
   Step 1: Review sample code comment style
   - Look at the "ACCEPTABLE COMMENT STYLE FROM SAMPLE CODE" section above
   - Note that sample code HAS docstrings, section headers, and descriptive print statements
   - These are ACCEPTABLE and should NOT be flagged
   
   Step 2: Compare generated code comments against sample
   - If generated code has similar comment style to sample code: ACCEPTABLE - do NOT flag
   - If generated code has method docstrings like sample: ACCEPTABLE - do NOT flag
   - If generated code has section headers like "# ====================": ACCEPTABLE - do NOT flag
   - If generated code has descriptive print statements: ACCEPTABLE - do NOT flag
   
   Step 3: Flag ONLY excessive/unnecessary comments
   - Flag ONLY if comments are clearly excessive compared to sample code style
   - Flag ONLY instructional comments that shouldn't be in production (e.g., "# Step 1:", "# Step 2:" repeated many times)
   - Flag ONLY TODO/FIXME comments that indicate incomplete code
   - Flag ONLY redundant comments that just repeat what code clearly shows
   - Flag ONLY long explanatory blocks that duplicate documentation
   
   CRITICAL RULES:
   ✗ Do NOT flag docstrings that explain method parameters and return values (sample code has these)
   ✗ Do NOT flag section headers like "# ==================== Linked Services ====================" (sample code has these)
   ✗ Do NOT flag descriptive print statements (sample code has these)
   ✗ Do NOT flag brief inline comments explaining "why" (sample code has these)
   ✓ Flag ONLY comments that are clearly excessive or instructional beyond sample code style
   ✓ Compare against sample code - if similar style, accept it

{agent1_context}
{agent2_context}
{agent2_mapping_context}
{agent3a_context}
{sample_context}

GENERATED CODE TO VALIDATE:
═══════════════════════════════════════════════════════════════════════════════
{generated_code[:8000]}

TASK:
═══════════════════════════════════════════════════════════════════════════════
Analyze the generated code for DEPLOYMENT-BLOCKING ISSUES ONLY.

VALIDATION PROCESS (FOLLOW EXACTLY):
1. Search the code thoroughly before flagging any issue
2. Verify columns/methods actually don't exist before reporting them missing
3. Check method names case-insensitively
4. Only flag SQL types in cast operations (critical deployment blocker)
5. CRITICAL: Check source output types:
   a. PREFERRED: All columns should be 'string' type in source output (like sample_code.py)
   b. ACCEPTABLE: Pre-typed columns if they work correctly
   c. FLAG: SQL types in source output (nvarchar, varchar, datetime2, etc.) - should use ADF types or string
6. CRITICAL: Check column name escaping for hyphens:
   a. Search for column names with hyphens in Agent 1/Agent 2 outputs (e.g., "columns-20", "columns-25")
   b. Verify these columns are escaped with {{}} in dataflow scripts (e.g., {{columns-25}})
   c. Check in: source output, select mapColumn, aggregate groupBy/first(), derive expressions, cast output
   d. FLAG: If column names with hyphens are NOT escaped with {{}} - this will cause deployment/runtime errors
7. CRITICAL: Check for empty derive() transformations:
   a. Search for pattern: "derive() ~>" in dataflow scripts
   b. If found, flag as deployment blocker - empty derive() causes "missing input stream" error
   c. Valid derive() must have expressions: "derive(Column = expression) ~>"
8. CRITICAL: Check for Load* names in transformations array:
   a. Search for pattern: "Transformation(name='Load" in code
   b. If found, flag as deployment blocker - Load* names are sinks, not transformations
   c. Load* names should ONLY appear in sinks array, never in transformations
9. CRITICAL: For comments:
   a. Compare generated code comments against sample code comment style
   b. If similar to sample code style: ACCEPTABLE - do NOT flag
   c. Only flag if clearly excessive or instructional beyond sample code
10. Accept code variations that are functionally correct
11. Don't flag style or structure differences (except for truly excessive comments)

OUTPUT FORMAT (JSON):
{{
  "is_valid": true,
  "issues": [],
  "feedback": "Code is valid and ready for deployment.",
  "validation_details": {{
    "code_structure": {{
      "has_all_methods": true,
      "missing_methods": [],
      "syntax_valid": true
    }},
    "deployment_blockers": {{
      "sql_types_in_cast": false,
      "sql_types_found": [],
      "syntax_errors": false,
      "missing_imports": false,
      "forbidden_operations": false
    }},
    "method_signatures": {{
      "deploy_complete_solution_valid": true,
      "has_sql_config": true,
      "has_blob_config": true,
      "signature_issues": []
    }},
    "code_cleanliness": {{
      "has_unnecessary_comments": false,
      "has_extra_information": false,
      "comments_to_remove": [],
      "cleanliness_issues": []
    }},
    "code_quality": {{
      "has_runtime_errors": false,
      "has_missing_methods": false
    }}
  }}
}}

CRITICAL INSTRUCTIONS:
1. BE LENIENT - Only flag actual deployment/runtime blockers
2. VERIFY FIRST - Search code to confirm issues exist before flagging
3. ACCEPT VARIATIONS - If code works but differs from expected, accept it
4. FOCUS ON SQL TYPES - This is the #1 deployment blocker to check (in cast operations)
5. CRITICAL: Check source output types - prefer all strings pattern, but accept pre-typed if they work
6. CRITICAL: Check column name escaping - column names with hyphens MUST be escaped with {{}} in dataflow scripts
7. CRITICAL: Check for empty derive() transformations - flag as deployment blocker if found
8. CRITICAL: Check for Load* names in transformations array - flag as deployment blocker if found
9. CRITICAL: For comments - Compare against sample code style, only flag if clearly excessive
10. If code matches sample code patterns, set is_valid to true with empty issues array
11. If code is functionally valid, set is_valid to true with empty issues array
12. Output ONLY valid JSON, nothing else"""
            
            # Try with JSON mode first
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a pragmatic code validator for Azure Data Factory Python SDK. You ONLY flag deployment-blocking issues that would cause runtime or deployment failures. You verify issues exist in code before flagging them. You accept code variations that work correctly. You are lenient and focus on actual errors, not style differences. CRITICAL: Compare against sample code references provided in prompt. Only flag issues that are clearly different from sample code patterns. Verify issues exist by comparing against sample code before flagging. CRITICAL: Check for empty derive() transformations - flag as deployment blocker if found (derive() with no expressions causes 'missing input stream' error). CRITICAL: Check for Load* names in transformations array - flag as deployment blocker if found (Load* names are sinks, not transformations, causes 'missing input stream' error). CRITICAL: For comments - Compare against sample code style, only flag if clearly excessive beyond sample code patterns. Output ONLY valid JSON."},
                        {"role": "user", "content": validation_prompt}
                    ],
                    temperature=0.1,
                    max_tokens=8000,
                    response_format={"type": "json_object"}
                )
            except Exception as e:
                # Fallback to regular response if JSON mode not supported
                print(f"JSON mode not supported in validation, trying without: {e}")
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a pragmatic code validator for Azure Data Factory Python SDK. You ONLY flag deployment-blocking issues that would cause runtime or deployment failures. You verify issues exist in code before flagging them. You accept code variations that work correctly. You are lenient and focus on actual errors, not style differences. CRITICAL: Compare against sample code references provided in prompt. Only flag issues that are clearly different from sample code patterns. Verify issues exist by comparing against sample code before flagging. CRITICAL: Check for empty derive() transformations - flag as deployment blocker if found (derive() with no expressions causes 'missing input stream' error). CRITICAL: Check for Load* names in transformations array - flag as deployment blocker if found (Load* names are sinks, not transformations, causes 'missing input stream' error). CRITICAL: For comments - Compare against sample code style, only flag if clearly excessive beyond sample code patterns. Output ONLY valid JSON."},
                        {"role": "user", "content": validation_prompt}
                    ],
                    temperature=0.1,
                    max_tokens=8000
                )
            
            validation_result = response.choices[0].message.content
            
            # Parse and validate JSON
            try:
                result_json = json.loads(validation_result)
                return result_json
            except json.JSONDecodeError:
                # Try to extract JSON from markdown or text
                json_match = re.search(r'\{.*\}', validation_result, re.DOTALL)
                if json_match:
                    try:
                        result_json = json.loads(json_match.group())
                        return result_json
                    except json.JSONDecodeError:
                        pass
                
                # If JSON parsing fails, return a basic validation result
                print("Warning: Agent 3C output is not valid JSON, returning basic validation")
                return {
                    "is_valid": False,
                    "issues": ["Validation agent output could not be parsed"],
                    "feedback": "The validation agent encountered an error. Please review the code manually.",
                    "validation_details": {}
                }
                
        except Exception as e:
            print(f"Error in Agent 3C validation: {type(e).__name__}: {e}")
            traceback.print_exc()
            return {
                "is_valid": False,
                "issues": [f"Validation error: {str(e)}"],
                "feedback": f"Validation agent encountered an error: {str(e)}. Please review the code manually.",
                "validation_details": {}
            }
    
    # ==================== AGENT 4: SINGLE TABLE CODE GENERATION ====================
    
    def generate_single_table_decision(self, table_name, schema, table_columns, csv_columns, 
                                       datatype_analysis=None, csv_filename=None):
        """
        Agent 4A: Generate decision JSON for a single table (simpler than Agent 3A, NO aggregate)
        
        Args:
            table_name: Name of the target table
            schema: Schema name
            table_columns: List of column names in the target table
            csv_columns: List of all CSV column names
            datatype_analysis: Optional datatype analysis from Agent 1
            csv_filename: CSV filename
            
        Returns:
            Decision JSON with activities, cast_columns, and csv_columns_mapping
        """
        try:
            if self.client is None:
                # Fallback decision without AI
                return self._create_fallback_single_table_decision(
                    table_name, table_columns, csv_columns, datatype_analysis
                )
            
            # Build column mapping context
            column_mapping_context = ""
            if table_columns and csv_columns:
                column_mapping_context = "\n\nCSV Columns:\n" + "\n".join([f"  - {col}" for col in csv_columns[:50]])
                column_mapping_context += f"\n\nTarget Table Columns ({table_name}):\n" + "\n".join([f"  - {col}" for col in table_columns])
            
            # Build datatype context
            datatype_context = ""
            if datatype_analysis and 'columns' in datatype_analysis:
                cast_recommendations = {}
                for col in table_columns:
                    # Try to find matching CSV column
                    matching_csv_col = None
                    for csv_col in csv_columns:
                        if csv_col.lower() == col.lower() or csv_col.replace('_', '').lower() == col.replace('_', '').lower():
                            matching_csv_col = csv_col
                            break
                    
                    if matching_csv_col and matching_csv_col in datatype_analysis['columns']:
                        col_info = datatype_analysis['columns'][matching_csv_col]
                        sql_type = col_info.get('sql_type', '')
                        if sql_type and sql_type.upper() not in ['NVARCHAR', 'VARCHAR', 'STRING', 'TEXT']:
                            cast_recommendations[col] = sql_type
                
                if cast_recommendations:
                    datatype_context = "\n\nRecommended Cast Types:\n" + "\n".join([f"  - {col}: {sql_type}" for col, sql_type in cast_recommendations.items()])
            
            user_prompt = f"""You are Agent 4A: Single Table Decision Agent.
Your task is to analyze a single table and decide which transformations are needed.

CRITICAL: This is a SIMPLE pipeline following sample_code.py pattern:
- Pattern: source → cast → sink (NO aggregate, NO derive unless absolutely necessary)
- All CSV columns are read as strings in the source
- Only cast columns that need type conversion (numeric, date, timestamp)
- Map CSV columns to table columns by exact name matching (case-insensitive)

TABLE INFORMATION:
- Table Name: {table_name}
- Schema: {schema}
- CSV File: {csv_filename or 'Unknown'}
{column_mapping_context}
{datatype_context}

TASK:
Analyze the table and CSV columns, then output a JSON decision object that will be used to generate sample_code.py-style code.

OUTPUT FORMAT (JSON):
{{
  "table_name": "{table_name}",
  "activities": ["select", "cast"],  // MUST be ["select", "cast"] only - NO aggregate, NO derive
  "cast_columns": {{
    "TableColumnName": "decimal(18,2)",  // Use table column names as keys
    "AnotherColumn": "integer",
    "DateColumn": "date",
    "TimestampColumn": "timestamp"
  }},
  "csv_columns_mapping": {{
    "CSV_Column_Name": "Table_Column_Name",  // Map ALL CSV columns that exist in table
    ...
  }}
}}

CRITICAL INSTRUCTIONS:
1. activities MUST be ["select", "cast"] only - NO aggregate, NO derive (unless date conversion absolutely required)
2. cast_columns: 
   - Use TABLE column names as keys (not CSV column names)
   - Only include columns that need type conversion (numeric, date, timestamp)
   - Use SQL types: "decimal(18,2)", "integer", "date", "timestamp"
   - If no casting needed, cast_columns can be empty {{}}
3. csv_columns_mapping: 
   - Map CSV column names (keys) to table column names (values)
   - Include ALL CSV columns that have corresponding table columns
   - Use exact name matching (case-insensitive)
   - This mapping will be used to generate the source output in dataflow script
4. Use datatype_analysis recommendations if provided
5. Ensure all table columns that exist in CSV are mapped in csv_columns_mapping

OUTPUT ONLY THE JSON OBJECT, nothing else."""
            
            # Try with JSON mode first
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are an expert in Azure Data Factory dataflow transformations. You analyze single table schemas and decide which simple transformations (select, cast) are needed for sample_code.py-style pipelines. Output ONLY valid JSON. NO aggregate operations. Map CSV columns to table columns accurately using exact name matching."},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.2,
                    max_tokens=16000,
                    response_format={"type": "json_object"}
                )
            except Exception as e:
                # Fallback to regular response if JSON mode not supported
                print(f"JSON mode not supported, trying without: {e}")
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are an expert in Azure Data Factory dataflow transformations. You analyze single table schemas and decide which simple transformations (select, cast) are needed for sample_code.py-style pipelines. Output ONLY valid JSON. NO aggregate operations. Map CSV columns to table columns accurately using exact name matching."},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.2,
                    max_tokens=16000
                )
            
            generated_decision = response.choices[0].message.content
            # Parse and validate JSON
            try:
                decision_json = json.loads(generated_decision)
                # Validate structure
                if 'table_name' in decision_json and 'activities' in decision_json:
                    # Ensure activities only contains select and cast
                    activities = decision_json.get('activities', [])
                    if 'aggregate' in activities:
                        activities = [a for a in activities if a != 'aggregate']
                        decision_json['activities'] = activities
                    return decision_json
            except json.JSONDecodeError:
                # Try to extract JSON from markdown or text
                json_match = re.search(r'\{.*\}', generated_decision, re.DOTALL)
                if json_match:
                    try:
                        decision_json = json.loads(json_match.group())
                        if 'table_name' in decision_json and 'activities' in decision_json:
                            return decision_json
                    except json.JSONDecodeError:
                        pass
            
            print("Warning: Agent 4A output is not valid JSON, using fallback")
            return self._create_fallback_single_table_decision(
                table_name, table_columns, csv_columns, datatype_analysis
            )
                
        except Exception as e:
            print(f"Error in Agent 4A decision generation: {type(e).__name__}: {e}")
            traceback.print_exc()
            return self._create_fallback_single_table_decision(
                table_name, table_columns, csv_columns, datatype_analysis
            )
    
    def _create_fallback_single_table_decision(self, table_name, table_columns, csv_columns, datatype_analysis=None):
        """Create fallback decision for single table"""
        cast_columns = {}
        csv_columns_mapping = {}
        
        # Map CSV columns to table columns
        for table_col in table_columns:
            # Try exact match first
            matching_csv_col = None
            for csv_col in csv_columns:
                if csv_col.lower() == table_col.lower():
                    matching_csv_col = csv_col
                    break
            
            # Try fuzzy match
            if not matching_csv_col:
                for csv_col in csv_columns:
                    if csv_col.replace('_', '').lower() == table_col.replace('_', '').lower():
                        matching_csv_col = csv_col
                        break
            
            if matching_csv_col:
                csv_columns_mapping[matching_csv_col] = table_col
                
                # Check if casting needed from datatype analysis
                if datatype_analysis and 'columns' in datatype_analysis:
                    if matching_csv_col in datatype_analysis['columns']:
                        sql_type = datatype_analysis['columns'][matching_csv_col].get('sql_type', '').upper()
                        if sql_type and 'INT' in sql_type:
                            cast_columns[table_col] = 'integer'
                        elif sql_type and 'DECIMAL' in sql_type:
                            cast_columns[table_col] = 'decimal(18,2)'
                        elif sql_type and 'DATE' in sql_type:
                            cast_columns[table_col] = 'date'
                        elif sql_type and 'TIME' in sql_type or 'DATETIME' in sql_type:
                            cast_columns[table_col] = 'timestamp'
        
        return {
            "table_name": table_name,
            "activities": ["select", "cast"] if cast_columns else ["select"],
            "cast_columns": cast_columns,
            "csv_columns_mapping": csv_columns_mapping
        }
    
    def generate_single_table_code_from_decision(self, decision, table_name, schema, azure_config,
                                                  csv_filename, blob_container='applicationdata', blob_folder='source',
                                                  csv_columns=None):
        """
        Agent 4B: Generate sample_code.py-style code for single table from Agent 4A decision
        
        Args:
            decision: Decision JSON from Agent 4A
            table_name: Name of the target table
            schema: Schema name
            azure_config: Azure configuration dictionary
            csv_filename: CSV filename
            blob_container: Blob container name
            blob_folder: Blob folder path
            csv_columns: Optional list of all CSV columns (if not provided, uses mapping keys)
            
        Returns:
            Complete Python class code following sample_code.py exact pattern
        """
        try:
            if self.client is None:
                raise ValueError("OpenAI client is not available")
            
            if not isinstance(decision, dict):
                raise ValueError("Decision must be a dictionary")
            
            # Cache sample_code.py reference (read once, reuse)
            if self._sample_code_reference_cache is None:
                sample_code_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'sample_code.py')
                if os.path.exists(sample_code_path):
                    with open(sample_code_path, 'r', encoding='utf-8') as f:
                        self._sample_code_reference_cache = f.read()[:2500]  # Only first 2500 chars needed
            sample_code_reference = self._sample_code_reference_cache or ""
            
            # Extract information from decision
            activities = decision.get('activities', ['select', 'cast'])
            cast_columns = decision.get('cast_columns', {})
            csv_columns_mapping = decision.get('csv_columns_mapping', {})
            
            # Build dataflow script
            # Get all CSV columns (prefer provided csv_columns, otherwise use mapping keys)
            if csv_columns:
                all_csv_columns = csv_columns
            else:
                all_csv_columns = list(csv_columns_mapping.keys()) if csv_columns_mapping else []
            
            # Extract only filename from csv_filename (remove folder path if present)
            csv_filename_clean = csv_filename
            if csv_filename and '/' in csv_filename:
                csv_filename_clean = csv_filename.split('/')[-1]
            elif csv_filename and '\\' in csv_filename:
                csv_filename_clean = csv_filename.split('\\')[-1]
            
            # Build source output
            source_output_lines = []
            for col in all_csv_columns:
                clean_col = col.replace(' ', '_').replace('-', '_').replace('.', '_')
                source_output_lines.append(f"      {clean_col} as string")
            
            source_output = ',\n'.join(source_output_lines)
            
            # Build cast output
            cast_output_lines = []
            for col, cast_type in cast_columns.items():
                clean_col = col.replace(' ', '_').replace('-', '_').replace('.', '_')
                # Map SQL types to ADF dataflow types (CRITICAL: ADF script does NOT support SQL-specific syntax)
                # Remove any SQL-specific syntax like COLLATE, varchar length, etc.
                cast_type_clean = cast_type.split()[0].lower() if cast_type else ''
                
                if 'decimal' in cast_type_clean or 'numeric' in cast_type_clean:
                    cast_output_lines.append(f"      {clean_col} as decimal(18,2)")
                elif 'int' in cast_type_clean or 'bigint' in cast_type_clean:
                    cast_output_lines.append(f"      {clean_col} as integer")
                elif 'date' in cast_type_clean and 'time' not in cast_type_clean:
                    cast_output_lines.append(f"      {clean_col} as date")
                elif 'time' in cast_type_clean or 'datetime' in cast_type_clean:
                    cast_output_lines.append(f"      {clean_col} as timestamp")
                else:
                    # Default to string if unknown type
                    cast_output_lines.append(f"      {clean_col} as string")
            
            cast_output = ',\n'.join(cast_output_lines)
            
            # Build dataflow script
            dataflow_script = f"""source(output(
{source_output}
),
allowSchemaDrift: true,
validateSchema: false,
ignoreNoFilesFound: false) ~> SourceCSV

"""
            
            if cast_columns:
                dataflow_script += f"""SourceCSV cast(output(
{cast_output}
),
errors: true) ~> CastTypes

CastTypes sink(allowSchemaDrift: true,
 validateSchema: false,
 deletable:false,
 insertable:true,
 updateable:false,
 upsertable:false,
 format: 'table',
 skipDuplicateMapInputs: true,
 skipDuplicateMapOutputs: true,
 errorHandlingOption: 'stopOnFirstError') ~> Load{table_name}"""
            else:
                dataflow_script += f"""SourceCSV sink(allowSchemaDrift: true,
 validateSchema: false,
 deletable:false,
 insertable:true,
 updateable:false,
 upsertable:false,
 format: 'table',
 skipDuplicateMapInputs: true,
 skipDuplicateMapOutputs: true,
 errorHandlingOption: 'stopOnFirstError') ~> Load{table_name}"""
            
            # Generate class name
            class_name = f"{table_name}CSVToSQLPipeline"
            
            # Build the complete code
            user_prompt = f"""Generate complete Python SDK code for Azure Data Factory following the EXACT pattern from sample_code.py.

REFERENCE CODE (sample_code.py) - STUDY THIS CAREFULLY:
{sample_code_reference[:3000]}...

TABLE INFORMATION:
- Table Name: {table_name}
- Schema: {schema}
- CSV File: {csv_filename_clean}  (NOTE: Use ONLY filename, NOT folder path in file_name parameter)
- Blob Container: {blob_container}
- Blob Folder: {blob_folder}

DATAFLOW SCRIPT (already generated - use this EXACTLY):
{dataflow_script}

AZURE CONFIGURATION:
{json.dumps(azure_config, indent=2)}

TASK:
Generate a complete Python class following sample_code.py EXACT structure. CRITICAL: Match sample_code.py patterns exactly.

═══════════════════════════════════════════════════════════════════════════════
CRITICAL REQUIREMENTS - READ CAREFULLY:
═══════════════════════════════════════════════════════════════════════════════

1. CLASS STRUCTURE:
   - Class name: {class_name}
   - __init__: Accept tenant_id, client_id, client_secret as parameters (NOT hardcoded)
   - Methods: get_credential(), create_sql_linked_service(), create_blob_storage_linked_service(), 
     create_source_csv_dataset(), create_sink_table_dataset(), create_dataflow(), create_pipeline(),
     deploy_complete_solution(), run_pipeline(), monitor_pipeline()

2. LINKED SERVICES (CRITICAL - Follow sample_code.py exactly):
   - MUST use SecureString(value=connection_string) wrapper
   - MUST wrap in LinkedServiceResource(properties=properties)
   - MUST validate result after creation - check that result.name matches expected name
   - MUST handle exceptions properly - if creation fails, raise exception immediately
   - Example from sample_code.py:
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
     print(f"✓ SQL Linked Service created: {{result.name}}")
     return result  # MUST return result
   - CRITICAL: If result is None or creation fails, raise exception - do NOT continue

3. SOURCE CSV DATASET (CRITICAL - Follow sample_code.py exactly):
   - MUST use DelimitedTextDataset (NOT AzureBlobDataset)
   - MUST use AzureBlobStorageLocation with container, folder_path, file_name
   - CRITICAL: file_name must be ONLY the filename (NOT include folder path)
   - If csv_filename contains path separators, extract only the filename part
   - MUST include: column_delimiter=',', encoding_name='UTF-8', first_row_as_header=True
   - MUST validate result after creation
   - Example from sample_code.py:
     properties = DelimitedTextDataset(
         linked_service_name=LinkedServiceReference(
             reference_name='BlobStorageLinkedService',
             type='LinkedServiceReference'  # MUST include type
         ),
         location=AzureBlobStorageLocation(
             container='{blob_container}',
             folder_path='{blob_folder}',  # Folder path here
             file_name='{csv_filename_clean}'    # ONLY filename, extract from csv_filename if it contains path
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
     print(f"✓ Source CSV Dataset created: {{result.name}}")
     return result  # MUST return result
   - CRITICAL: Verify linked service 'BlobStorageLinkedService' exists before creating dataset

4. SINK TABLE DATASET (CRITICAL - Follow sample_code.py exactly):
   - MUST use separate schema and table parameters (NOT table_name='schema.table')
   - MUST validate result after creation
   - CRITICAL: Verify linked service 'SQLLinkedService' exists before creating dataset
   - Example from sample_code.py:
     properties = AzureSqlTableDataset(
         linked_service_name=LinkedServiceReference(
             reference_name='SQLLinkedService',
             type='LinkedServiceReference'  # MUST include type
         ),
         schema='{schema}',      # Separate parameter
         table='{table_name}'    # Separate parameter
     )
     dataset = DatasetResource(properties=properties)
     result = self.client.datasets.create_or_update(
         self.resource_group,
         self.factory_name,
         name,
         dataset
     )
     print(f"✓ Sink Table Dataset created: {{result.name}}")
     return result  # MUST return result

5. DATAFLOW (CRITICAL - This is the MOST IMPORTANT):
   - MUST use MappingDataFlow (NOT DataFlow)
   - MUST include script parameter with the provided dataflow script
   - MUST use simple Transformation(name='CastTypes') references (NOT DataFlowTransformation)
   - MUST use DatasetReference with type='DatasetReference'
   - CRITICAL: In the cast() transformation in script, use ONLY basic ADF types:
     * integer (NOT int, NOT bigint)
     * decimal(18,2) (NOT decimal with other precision, NOT numeric)
     * date (NOT datetime for dates)
     * timestamp (for datetime/timestamp)
     * string (for text)
     * DO NOT use SQL-specific syntax like COLLATE, varchar(50), etc.
   - CRITICAL: Verify source dataset 'Source{{table_name}}CSV' and sink dataset 'Sink{{table_name}}' exist before creating dataflow
   - MUST validate result after creation
   - Example from sample_code.py:
     script = \"\"\"{dataflow_script}\"\"\"
     
     dataflow_properties = MappingDataFlow(  # MUST be MappingDataFlow
         sources=[
             DataFlowSource(
                 name='SourceCSV',
                 dataset=DatasetReference(
                     reference_name='Source{{table_name}}CSV',
                     type='DatasetReference'  # MUST include type
                 )
             )
         ],
         sinks=[
             DataFlowSink(
                 name='Load{{table_name}}',
                 dataset=DatasetReference(
                     reference_name='Sink{{table_name}}',
                     type='DatasetReference'  # MUST include type
                 )
             )
         ],
         transformations=[
             Transformation(name='CastTypes')  # Simple reference, NOT DataFlowTransformation
         ],
         script=script  # CRITICAL: script parameter is REQUIRED!
     )
     dataflow = DataFlowResource(properties=dataflow_properties)
     result = self.client.data_flows.create_or_update(
         self.resource_group,
         self.factory_name,
         name,
         dataflow
     )
     print(f"✓ Data Flow created: {{result.name}}")
     return result  # MUST return result

6. PIPELINE (CRITICAL - Follow sample_code.py exactly):
   - MUST use ExecuteDataFlowActivity (NOT DataFlowActivity)
   - MUST include ActivityPolicy with timeout, retry settings
   - MUST include compute configuration
   - MUST include trace_level
   - Pipeline name should follow pattern: {{table_name}}CSVToSQLPipeline (e.g., FactSalesCSVToSQLPipeline)
   - Activity name should be descriptive: Load{{table_name}}DataFlowActivity
   - CRITICAL: The variable name in create_pipeline() MUST be 'name' (not 'pipeline_name' or any other name)
   - CRITICAL: The pipeline name value MUST be exactly: '{{table_name}}CSVToSQLPipeline' (must end with 'Pipeline')
   - CRITICAL: The pipeline name MUST be unique and different from linked service names (SQLLinkedService, BlobStorageLinkedService)
   - CRITICAL: Verify dataflow 'Load{{table_name}}DataFlow' exists before creating pipeline
   - MUST validate result after creation
   - CRITICAL: The pipeline MUST reference the dataflow by the EXACT name used when creating it
   - Example from sample_code.py:
     def create_pipeline(self):
         \"\"\"Create pipeline with single data flow activity\"\"\"
         name = '{{table_name}}CSVToSQLPipeline'  # CRITICAL: Variable must be 'name', value must end with 'Pipeline'
         print(f"Creating Pipeline: {{name}}...")
         
         dataflow_activity = ExecuteDataFlowActivity(  # MUST be ExecuteDataFlowActivity
             name='Load{{table_name}}DataFlowActivity',
             policy=ActivityPolicy(
                 timeout='0.12:00:00',
                 retry=0,
                 retry_interval_in_seconds=30,
                 secure_output=False,
                 secure_input=False
             ),
             data_flow=DataFlowReference(
                 reference_name='Load{{table_name}}DataFlow',  # MUST match the dataflow name exactly
                 type='DataFlowReference'  # MUST include type
             ),
             compute=ExecuteDataFlowActivityTypePropertiesCompute(
                 compute_type='General',
                 core_count=8
             ),
             trace_level='Fine'
         )
         
         pipeline = PipelineResource(
             description='Pipeline to load {{table_name}} data from CSV to SQL',
             activities=[dataflow_activity]  # Direct activities list
         )
         result = self.client.pipelines.create_or_update(
             self.resource_group,
             self.factory_name,
             name,  # CRITICAL: Use the 'name' variable defined above
             pipeline
         )
         print(f"✓ Pipeline created: {{result.name}}")
         return result  # MUST return result
   - CRITICAL: The pipeline name variable MUST be defined at the start of create_pipeline() method
   - CRITICAL: Do NOT use 'pipeline_name' as variable name - use 'name' to match sample_code.py exactly

7. METHOD IMPLEMENTATIONS:
   - ALL create methods MUST return the result
   - ALL create methods MUST have print statements for success
   - ALL create methods MUST validate that result is not None
   - ALL create methods MUST validate that result.name matches expected name
   - ALL create methods MUST handle exceptions and raise them (do NOT swallow errors)
   - If creation fails, method MUST raise exception immediately (do NOT return None)
   - CRITICAL: Wrap create_or_update calls in try-except to catch and re-raise exceptions with context
   - Example: 
     try:
         result = self.client.linked_services.create_or_update(
             self.resource_group,
             self.factory_name,
             name,
             linked_service
         )
         if result is None:
             raise Exception(f"Failed to create linked service: {{name}} - result is None")
         if result.name != name:
             raise Exception(f"Linked service name mismatch: expected {{name}}, got {{result.name}}")
         print(f"✓ SQL Linked Service created: {{result.name}}")
         return result
     except Exception as e:
         print(f"✗ Failed to create linked service {{name}}: {{str(e)}}")
         raise  # Re-raise to stop deployment
   - Include try-except error handling in deploy_complete_solution()
   - CRITICAL: Each create method should validate success before returning
   - CRITICAL: If any create method fails, the exception MUST propagate to stop deployment

8. RUN_PIPELINE (CRITICAL - Follow sample_code.py exactly):
   - MUST accept parameters=None parameter
   - MUST use self.client.pipelines.create_run() (NOT pipelines.run())
   - MUST include try-except error handling
   - MUST print success message with run_id
   - MUST return run_id: return run_response.run_id
   - MUST return None on error
   - Example from sample_code.py:
     def run_pipeline(self, parameters=None):
         print("Starting pipeline execution...")
         try:
             run_response = self.client.pipelines.create_run(  # MUST be create_run
                 self.resource_group,
                 self.factory_name,
                 'PipelineName',
                 parameters=parameters or {{}}
             )
             print(f"✓ Pipeline started successfully")
             print(f"  Run ID: {{run_response.run_id}}")
             return run_response.run_id
         except Exception as e:
             print(f"✗ Failed to start pipeline: {{str(e)}}")
             return None

9. MONITOR_PIPELINE (CRITICAL - Follow sample_code.py exactly):
   - MUST accept run_id and check_interval=10 parameters
   - MUST check if run_id is None and return None if invalid
   - MUST include proper monitoring loop with status checking
   - MUST format timestamps: time.strftime('%Y-%m-%d %H:%M:%S')
   - MUST handle KeyboardInterrupt exception
   - MUST include detailed status messages for Succeeded/Failed/Cancelled
   - MUST return status when pipeline completes
   - Example from sample_code.py:
     def monitor_pipeline(self, run_id, check_interval=10):
         if not run_id:
             print("No valid run ID provided")
             return None
         print(f"\\nMonitoring pipeline run: {{run_id}}")
         print("-" * 80)
         try:
             while True:
                 pipeline_run = self.client.pipeline_runs.get(...)
                 status = pipeline_run.status
                 timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
                 print(f"[{{timestamp}}] Status: {{status}}")
                 if status in ['Succeeded', 'Failed', 'Cancelled']:
                     print("-" * 80)
                     if status == 'Succeeded':
                         print("✓ Pipeline execution completed successfully!")
                     elif status == 'Failed':
                         print("✗ Pipeline execution failed.")
                     else:
                         print("⚠ Pipeline execution was cancelled.")
                     return status
                 time.sleep(check_interval)
         except KeyboardInterrupt:
             print("\\n⚠ Monitoring interrupted by user")
             return None
         except Exception as e:
             print(f"✗ Error during monitoring: {{str(e)}}")
             return None

10. DEPLOY_COMPLETE_SOLUTION (CRITICAL - Follow sample_code.py exactly):
    - MUST include docstring: \"\"\"Deploy complete simple CSV to SQL pipeline\"\"\"
    - MUST include structured output with step-by-step messages
    - MUST use try-except with traceback on error
    - MUST print section headers with "=" and "-" separators
    - MUST print success message at end with "✓ DEPLOYMENT COMPLETED SUCCESSFULLY!"
    - MUST include "Resources Created:" section at the end (even if empty list - sample_code.py has this)
    - CRITICAL: Each create method MUST complete successfully before moving to next step
    - CRITICAL: If any create method fails, deployment MUST stop immediately and raise exception
    - CRITICAL: Do NOT continue if linked services fail to create - they are required for datasets
    - CRITICAL: Do NOT continue if datasets fail to create - they are required for dataflow
    - CRITICAL: Do NOT continue if dataflow fails to create - it is required for pipeline
    - CRITICAL: The try-except MUST use 'import traceback' and 'traceback.print_exc()' BEFORE raising
    - CRITICAL: After "Resources Created:", leave a blank line (sample_code.py format)
    - Example from sample_code.py:
      def deploy_complete_solution(self):
          \"\"\"Deploy complete simple CSV to SQL pipeline\"\"\"
          print("=" * 80)
          print("DEPLOYING SIMPLE CSV TO SQL PIPELINE")
          print("=" * 80)
          print()
          try:
              print("Step 1: Creating Linked Services")
              print("-" * 80)
              self.create_sql_linked_service()  # MUST succeed before continuing
              self.create_blob_storage_linked_service()  # MUST succeed before continuing
              print()
              
              print("Step 2: Creating Datasets")
              print("-" * 80)
              self.create_source_csv_dataset()  # MUST succeed before continuing
              self.create_sink_table_dataset()  # MUST succeed before continuing
              print()
              
              print("Step 3: Creating Data Flow")
              print("-" * 80)
              self.create_dataflow()  # MUST succeed before continuing
              print()
              
              print("Step 4: Creating Pipeline")
              print("-" * 80)
              self.create_pipeline()  # MUST succeed before continuing
              print()
              
              print("=" * 80)
              print("✓ DEPLOYMENT COMPLETED SUCCESSFULLY!")
              print("=" * 80)
              print()
              print("Resources Created:")
              
          except Exception as e:
              print(f"✗ Deployment failed: {{str(e)}}")
              import traceback
              traceback.print_exc()
              raise  # MUST raise to stop execution - do NOT swallow exceptions

11. MAIN FUNCTION:
    - MUST follow sample_code.py main() structure exactly
    - MUST define credentials as constants in main()
    - MUST pass credentials to class constructor
    - MUST include optional user input for running pipeline
    - MUST include optional user input for monitoring
    - Example from sample_code.py:
      def main():
          TENANT_ID = '...'
          CLIENT_ID = '...'
          CLIENT_SECRET = '...'
          SUBSCRIPTION_ID = '...'
          RESOURCE_GROUP = '...'
          FACTORY_NAME = '...'
          LOCATION = '...'
          
          pipeline_manager = ClassName(...)
          pipeline_manager.deploy_complete_solution()
          
          print("\\nDeployment complete!")
          user_input = input("Do you want to run the pipeline now? (yes/no): ")
          if user_input.lower() in ['yes', 'y']:
              run_id = pipeline_manager.run_pipeline()
              if run_id:
                  monitor_input = input("\\nDo you want to monitor? (yes/no): ")
                  if monitor_input.lower() in ['yes', 'y']:
                      status = pipeline_manager.monitor_pipeline(run_id)
                      print(f"\\nFinal Status: {{status}}")

═══════════════════════════════════════════════════════════════════════════════
COMMON MISTAKES TO AVOID (These will cause deployment failures):
═══════════════════════════════════════════════════════════════════════════════

❌ DO NOT use AzureBlobDataset - use DelimitedTextDataset
❌ DO NOT use DataFlow - use MappingDataFlow
❌ DO NOT use DataFlowActivity - use ExecuteDataFlowActivity
❌ DO NOT use table_name='schema.table' - use separate schema and table
❌ DO NOT use object-based dataflow (output=Output(...)) - use script parameter
❌ DO NOT use DataFlowTransformation - use simple Transformation(name=...)
❌ DO NOT hardcode credentials in __init__ - accept as parameters
❌ DO NOT forget SecureString wrapper for connection strings
❌ DO NOT forget type='LinkedServiceReference' and type='DatasetReference'
❌ DO NOT forget to return values from create methods
❌ DO NOT include folder path in file_name - use ONLY filename
❌ DO NOT use SQL-specific syntax in cast (like COLLATE, varchar(50)) - use basic ADF types only
❌ DO NOT use pipelines.run() - use pipelines.create_run()
❌ DO NOT skip structured deployment messages - include step-by-step output
❌ DO NOT skip detailed monitoring - include timestamps and status messages
❌ DO NOT continue deployment if any resource creation fails
❌ DO NOT create pipeline if dataflow doesn't exist
❌ DO NOT create dataflow if datasets don't exist
❌ DO NOT create datasets if linked services don't exist
❌ DO NOT return None from create methods - raise exception on failure

✅ DO use script parameter in MappingDataFlow
✅ DO validate each resource is created successfully before proceeding
✅ DO include "✓ DEPLOYMENT COMPLETED SUCCESSFULLY!" message
✅ DO include "Resources Created:" section at end
✅ DO raise exceptions immediately if resource creation fails
✅ DO use DelimitedTextDataset with AzureBlobStorageLocation
✅ DO use ExecuteDataFlowActivity with ActivityPolicy, compute, trace_level
✅ DO use separate schema and table parameters
✅ DO use SecureString wrapper
✅ DO include all type parameters
✅ DO return values from all methods
✅ DO include error handling
✅ DO extract only filename from csv_filename (remove folder path if present)
✅ DO use basic ADF types in cast: integer, decimal(18,2), date, timestamp, string
✅ DO use pipelines.create_run() (NOT pipelines.run())
✅ DO include structured deployment messages with step headers
✅ DO include detailed monitoring with timestamps and status messages

Generate ONLY the Python code, starting with the class definition. Follow sample_code.py EXACTLY."""
            
            system_prompt = """You generate complete, working Python SDK code for Azure Data Factory following the test004.py pattern EXACTLY.

CRITICAL RULES - These are MANDATORY (deviations will cause deployment failures):
1. Use MappingDataFlow with script parameter (NOT DataFlow with object-based structure)
2. Use DelimitedTextDataset with AzureBlobStorageLocation (NOT AzureBlobDataset)
3. Use SecureString(value=...) wrapper for ALL connection strings
4. Use ExecuteDataFlowActivity with ActivityPolicy, compute, trace_level (NOT DataFlowActivity)
5. Use separate schema and table parameters for AzureSqlTableDataset (NOT table_name='schema.table')
6. Use simple Transformation(name=...) references (NOT DataFlowTransformation with type='DerivedColumn')
7. Include type='LinkedServiceReference' and type='DatasetReference' in all references
8. Return values from ALL create methods
9. Accept credentials as parameters in __init__ (NOT hardcode them)
10. Include proper error handling in deploy_complete_solution() with structured step messages
11. Include proper monitoring logic in monitor_pipeline() with run_id parameter, timestamps, and detailed status
12. Use pipelines.create_run() (NOT pipelines.run())
13. Extract ONLY filename from csv_filename for file_name parameter (remove folder path if present)
14. In dataflow script cast(), use ONLY basic ADF types: integer, decimal(18,2), date, timestamp, string
    - DO NOT use SQL-specific syntax like COLLATE, varchar(50), etc.
15. Follow test004.py structure EXACTLY - every method, every parameter, every pattern, every print statement
16. RESOURCE NAME MATCHING (CRITICAL):
    - Linked service names MUST be EXACTLY: 'SQLLinkedService' and 'BlobStorageLinkedService'
    - Dataset names MUST match exactly: 'Source{{table_name}}CSV' and 'Sink{{table_name}}'
    - Dataflow name MUST match exactly: 'Load{{table_name}}DataFlow'
    - Pipeline name MUST match exactly: '{{table_name}}CSVToSQLPipeline'
    - DataFlowReference reference_name MUST match the dataflow name EXACTLY
    - DatasetReference reference_name MUST match the dataset names EXACTLY
    - LinkedServiceReference reference_name MUST match the linked service names EXACTLY
    - CRITICAL: Any mismatch in names will cause "Entity not found" errors when running pipeline
    - CRITICAL: When creating pipeline, the DataFlowReference must use the EXACT same name as the dataflow created
    - CRITICAL: When creating dataflow, the DatasetReference names must match EXACT dataset names created

17. DEPLOYMENT ORDER AND VALIDATION (CRITICAL):
    - Resources MUST be created in this exact order:
      1. Linked Services (SQLLinkedService, BlobStorageLinkedService)
      2. Datasets (Source{{table_name}}CSV, Sink{{table_name}})
      3. Dataflow (Load{{table_name}}DataFlow)
      4. Pipeline ({{table_name}}CSVToSQLPipeline)
    - Each step MUST complete successfully before moving to next
    - If ANY step fails, deployment MUST stop and raise exception
    - Do NOT continue deployment if previous step failed
    - CRITICAL: The deployment MUST complete ALL steps before pipeline can be run
    - CRITICAL: If deployment fails partway through, resources may be in inconsistent state

The generated code MUST be deployable and executable. Any deviation from test004.py patterns will cause deployment failures."""
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                max_tokens=16000
            )
            
            generated_code = response.choices[0].message.content
            
            # Extract code from markdown if present
            if '```' in generated_code:
                code_pattern = r'```(?:python)?\s*\n(.*?)\n```'
                matches = re.findall(code_pattern, generated_code, re.DOTALL)
                if matches:
                    generated_code = matches[0].strip()
            
            if not generated_code or len(generated_code.strip()) == 0:
                raise ValueError("Generated code is empty")
            
            return generated_code
            
        except Exception as e:
            error_msg = f"Error in Agent 4B code generation: {type(e).__name__}: {str(e)}"
            print(error_msg)
            traceback.print_exc()
            raise Exception(error_msg) from e
    
    def generate_python_sdk_code(self, csv_analysis, datatype_analysis, destination_tables, azure_config, 
                                 csv_data=None, blob_container=None, blob_folder=None, csv_filename=None,
                                 stream_container=None):
        """
        Agent 3: Orchestrates Agent 3A and 3B to generate complete Python SDK code.
        This is the main entry point for backwards compatibility.
        
        Args:
            csv_analysis: Agent 1 output with fact/dimension structure
            datatype_analysis: Agent 2 output with SQL type recommendations
            destination_tables: Target tables dictionary
            azure_config: Azure configuration
            csv_data: DataFrame with CSV data
            blob_container: Blob container name (default: 'applicationdata')
            blob_folder: Blob folder path (default: 'source')
            csv_filename: Full CSV file path from frontend (e.g., 'source/Sunrise_Medical_Center.csv')
            stream_container: Optional Streamlit container for displaying streaming response (for code generation)
        
        Returns:
            dict with keys:
                - code: Generated Python SDK code (string)
                - validation_result: Agent 3C validation result (dict with is_valid, issues, feedback, validation_details)
                - attempt_count: Number of validation attempts made (int)
                - final_validation_status: Whether validation passed or failed (bool)
                - agent3a_decision: Agent 3A decision JSON (dict, optional)
        """
        try:
            if csv_analysis is None:
                raise ValueError("CSV analysis (Agent 1 output) is required")
            if datatype_analysis is None:
                raise ValueError("Data type analysis (Agent 2 output) is required")
            if not destination_tables:
                raise ValueError("At least one destination table must be selected")
            
            # Extract folder_path and file_name from csv_filename if provided
            # csv_filename format: 'source/Sunrise_Medical_Center.csv' or 'Sunrise_Medical_Center.csv'
            extracted_folder_path = blob_folder or 'source'
            extracted_file_name = 'healthcare_data_sample.csv'  # default
            
            if csv_filename:
                # Handle both 'source/file.csv' and 'file.csv' formats
                if '/' in csv_filename:
                    parts = csv_filename.split('/', 1)
                    extracted_folder_path = parts[0] if parts[0] else blob_folder or 'source'
                    extracted_file_name = parts[1] if len(parts) > 1 else csv_filename
                elif '\\' in csv_filename:
                    parts = csv_filename.split('\\', 1)
                    extracted_folder_path = parts[0] if parts[0] else blob_folder or 'source'
                    extracted_file_name = parts[1] if len(parts) > 1 else csv_filename
                else:
                    # Just filename, use provided blob_folder or default
                    extracted_file_name = csv_filename
                    extracted_folder_path = blob_folder or 'source'
            
            # First, try Agent 3A to generate decision JSON
            agent3a_decision = self.generate_pipeline_prompt(
                csv_analysis, datatype_analysis, destination_tables, azure_config,
                csv_data, blob_container, blob_folder
            )
            
            # Build agent2_mapping structure from csv_analysis (similar to agent2_datatype_mapping.json)
            agent2_mapping = None
            if csv_analysis:
                fact_columns = csv_analysis.get('fact_columns', [])
                dimensions = csv_analysis.get('dimensions', {})
                fact_table_name = csv_analysis.get('fact_table', {}).get('name', 'FactVisit')
                
                agent2_mapping = {
                    "fact_table": {
                        "name": fact_table_name,
                        "fact_columns": fact_columns
                    },
                    "dimensions": dimensions,
                    "foreign_keys": csv_analysis.get('foreign_keys', {})
                }
            
            # If Agent 3A succeeded, use Agent 3B to generate code from decision with validation loop
            if agent3a_decision:
                print("Agent 3A: Decision logic generated successfully")
                
                # Read sample code for validation
                sample_code_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'templates', 'sample_code.py')
                sample_code = ""
                if os.path.exists(sample_code_path):
                    with open(sample_code_path, 'r', encoding='utf-8') as f:
                        sample_code = f.read()
                
                # Feedback loop with validation (max 3 attempts)
                max_retries = 2
                validation_feedback = None
                final_validation_result = None
                
                for attempt in range(max_retries):
                    print(f"\n{'='*80}")
                    print(f"CODE GENERATION ATTEMPT {attempt + 1}/{max_retries}")
                    if validation_feedback:
                        print(f"Addressing validation feedback from previous attempt...")
                    print(f"{'='*80}\n")
                    
                    # Generate code with Agent 3B (with feedback if available)
                    # Only stream on first attempt to avoid cluttering UI with retries
                    code_stream_container = stream_container if attempt == 0 else None
                    code = self.generate_python_sdk_code_from_prompt(
                        agent3a_decision,
                        csv_analysis=csv_analysis,
                        datatype_analysis=datatype_analysis,
                        agent2_mapping=agent2_mapping,
                        csv_filename=csv_filename,
                        blob_container=blob_container or 'applicationdata',
                        blob_folder=extracted_folder_path,
                        file_name=extracted_file_name,
                        validation_feedback=validation_feedback,
                        stream_container=code_stream_container
                    )
                    print(f"Agent 3B: Code generated (attempt {attempt + 1})")
                    
                    # Validate code with Agent 3C
                    print("Agent 3C: Validating generated code...")
                    validation_result = self.validate_generated_code(
                        generated_code=code,
                        agent3a_decision=agent3a_decision,
                        csv_analysis=csv_analysis,
                        datatype_analysis=datatype_analysis,
                        agent2_mapping=agent2_mapping,
                        sample_code=sample_code
                    )
                    
                    # Track the validation result
                    final_validation_result = validation_result
                    
                    if validation_result.get('is_valid', False):
                        print("✅ Agent 3C: Code validation PASSED!")
                        print("Code is ready for deployment.")
                        return {
                            "code": code,
                            "validation_result": validation_result,
                            "attempt_count": attempt + 1,
                            "final_validation_status": True,
                            "agent3a_decision": agent3a_decision
                        }
                    else:
                        issues = validation_result.get('issues', [])
                        feedback = validation_result.get('feedback', 'No specific feedback provided')
                        
                        print(f"❌ Agent 3C: Code validation FAILED (attempt {attempt + 1}/{max_retries})")
                        print(f"Issues found: {len(issues)}")
                        for i, issue in enumerate(issues[:5], 1):  # Show first 5 issues
                            print(f"  {i}. {issue}")
                        if len(issues) > 5:
                            print(f"  ... and {len(issues) - 5} more issues")
                        
                        # Prepare feedback for next iteration
                        validation_feedback = feedback
                        
                        # If this is not the last attempt, regenerate with feedback
                        if attempt < max_retries - 1:
                            print(f"\n🔄 Regenerating code with validation feedback...")
                            # Also regenerate Agent 3A decision with feedback
                            agent3a_decision = self.generate_pipeline_prompt(
                                csv_analysis, datatype_analysis, destination_tables, azure_config,
                                csv_data, blob_container, blob_folder, validation_feedback=validation_feedback
                            )
                            if not agent3a_decision:
                                print("⚠️ Agent 3A failed to regenerate decision, using previous decision")
                        else:
                            print(f"\n⚠️ Maximum retries reached. Returning code with validation issues.")
                            print("You may need to review and fix the code manually.")
                            return {
                                "code": code,
                                "validation_result": validation_result,
                                "attempt_count": attempt + 1,
                                "final_validation_status": False,
                                "agent3a_decision": agent3a_decision
                            }
                
                # If we get here, all retries exhausted but validation still failed
                # This shouldn't happen, but handle it defensively
                if final_validation_result is None:
                    final_validation_result = {
                        "is_valid": False,
                        "issues": ["Validation error - no validation result available"],
                        "feedback": "Code generation completed but validation result is missing.",
                        "validation_details": {}
                    }
                
                print("⚠️ Code generation completed but validation failed after all retries.")
                return {
                    "code": code,
                    "validation_result": final_validation_result,
                    "attempt_count": max_retries,
                    "final_validation_status": False,
                    "agent3a_decision": agent3a_decision
                }
            else:
                # Fallback to direct generation if Agent 3A fails
                print("Agent 3A: Decision generation failed, falling back to direct code generation")
                csv_columns = csv_data.columns.tolist() if csv_data is not None else []
                
                fact_columns = csv_analysis.get('fact_columns', [])
                raw_dimensions = csv_analysis.get('dimensions', {})
                dimensions = self._normalize_dimensions(raw_dimensions)
                foreign_keys = csv_analysis.get('foreign_keys', {})
                
                column_types = {}
                if datatype_analysis and 'columns' in datatype_analysis:
                    column_types = datatype_analysis['columns']
                
                fact_tables = []
                dim_tables = []
                table_schemas = {}
                for table_key, table_info in destination_tables.items():
                    if '.' in table_key:
                        schema, table = table_key.split('.', 1)
                        table_schemas[table] = schema
                        tl = table.lower()
                        if tl.startswith('fact') or tl.startswith('ft_'):
                            fact_tables.append((table, schema))
                        elif tl.startswith('dim') or tl.startswith('dim_'):
                            dim_tables.append((table, schema))
                        else:
                            matched = False
                            for dim_name in dimensions.keys():
                                # Validate dim_name is a string
                                if not isinstance(dim_name, str):
                                    continue
                                if dim_name.replace('Dim', '').lower() in tl:
                                    dim_tables.append((table, schema))
                                    matched = True
                                    break
                            if not matched:
                                fact_tables.append((table, schema))
                
                context_keyword = self._derive_context_keyword(csv_columns, fact_columns, dimensions)
                
                code = self._generate_complete_sdk_code(
                    context_keyword=context_keyword,
                    csv_columns=csv_columns,
                    fact_columns=fact_columns,
                    dimensions=dimensions,
                    foreign_keys=foreign_keys,
                    column_types=column_types,
                    fact_tables=fact_tables,
                    dim_tables=dim_tables,
                    table_schemas=table_schemas,
                    azure_config=azure_config,
                    blob_container=blob_container or 'applicationdata',
                    blob_folder=blob_folder or 'source'
                )
                if not code or len(code.strip()) == 0:
                    raise ValueError("Generated code is empty")
                
                # Fallback case: no validation performed, return code with default validation result
                return {
                    "code": code,
                    "validation_result": {
                        "is_valid": False,
                        "issues": ["Validation skipped - Agent 3A failed, using fallback generation"],
                        "feedback": "Code generated using fallback method. Validation was not performed.",
                        "validation_details": {}
                    },
                    "attempt_count": 0,
                    "final_validation_status": False,
                    "agent3a_decision": None
                }
        except Exception as e:
            error_msg = f"Error in Agent 3 code generation: {type(e).__name__}: {str(e)}"
            print(error_msg)
            traceback.print_exc()
            raise Exception(error_msg) from e
    
    def generate_python_sdk_code_v3_training(self, csv_analysis, datatype_analysis, destination_tables, azure_config,
                                             csv_data=None, blob_container=None, blob_folder=None):
        """Agent 3: Training-based code generation with retry logic"""
        max_retries = 2
        
        for attempt in range(max_retries):
            try:
                print(f"\n{'='*80}")
                print(f"CODE GENERATION ATTEMPT {attempt + 1}/{max_retries}")
                print(f"{'='*80}\n")
                
                if self.client is None:
                    raise ValueError("OpenAI client is not initialized")
                
                # ... existing validation code ...
                
                # Prepare dimensions info for validation
                dimensions = csv_analysis.get('dimensions', {})
                dimensions = self._normalize_dimensions(dimensions)
                dimension_count = len(dimensions)
                
                print(f"Expected components:")
                print(f"  - Dimensions: {dimension_count} ({', '.join(dimensions.keys())})")
                print(f"  - SELECT transformations: {dimension_count + 1}")
                print(f"  - AGGREGATE transformations: {dimension_count}")
                print(f"  - LOAD sinks: {dimension_count + 1}")
                
                # Enhanced user prompt with explicit examples
                user_prompt = f"""Generate COMPLETE Python SDK code for Azure Data Factory pipeline.

CRITICAL INSTRUCTION - READ THIS FIRST:
════════════════════════════════════════════════════════════════════════════
You MUST generate transformations for ALL {dimension_count} dimensions BEFORE the fact table.

Agent 1 detected these dimensions:
{json.dumps(list(dimensions.keys()), indent=2)}

For EACH dimension above, you MUST include in the dataflow script:
1. StagingSource select(mapColumn(...)) ~> SelectDimXXX
2. SelectDimXXX aggregate(groupBy(pk), ...) ~> AggregateDimXXX  
3. AggregateDimXXX sink(...) ~> LoadDimXXX

ONLY AFTER all {dimension_count} dimensions, add the fact table transformation.

Expected script structure (example for Hospital with 5 dimensions):

script = \"\"\"source(...) ~> StagingSource

StagingSource select(...) ~> SelectDimDate
SelectDimDate aggregate(...) ~> AggregateDimDate
AggregateDimDate sink(...) ~> LoadDimDate

StagingSource select(...) ~> SelectDimDoctor
SelectDimDoctor aggregate(...) ~> AggregateDimDoctor
AggregateDimDoctor sink(...) ~> LoadDimDoctor

StagingSource select(...) ~> SelectDimHospital
SelectDimHospital aggregate(...) ~> AggregateDimHospital
AggregateDimHospital sink(...) ~> LoadDimHospital

StagingSource select(...) ~> SelectDimMedication
SelectDimMedication aggregate(...) ~> AggregateDimMedication
AggregateDimMedication sink(...) ~> LoadDimMedication

StagingSource select(...) ~> SelectDimPatient
SelectDimPatient aggregate(...) ~> AggregateDimPatient
AggregateDimPatient sink(...) ~> LoadDimPatient

StagingSource select(...) ~> SelectFactVisit
SelectFactVisit sink(...) ~> LoadFactVisit\"\"\"

Your generated script MUST follow this exact pattern with ALL dimensions.
════════════════════════════════════════════════════════════════════════════

AGENT 1 OUTPUT (Full):
{json.dumps(csv_analysis, indent=2)}

AGENT 2 OUTPUT:
{json.dumps(datatype_analysis, indent=2)}

DESTINATION TABLES:
{json.dumps(destination_tables, indent=2)}

AZURE CONFIG:
{json.dumps(azure_config, indent=2)}

BLOB STORAGE:
  Container: {blob_container or 'applicationdata'}
  Folder: {blob_folder or 'source'}

Generate the COMPLETE Python file with:
1. ALL dimension transformations in dataflow script
2. create_dimension_datasets() method
3. Complete resource_names dictionary with Neccessory resources names as per agent3a_decision
4. Proper transformations and sinks lists

VERIFY before completing:
- Script has {dimension_count} × 3 = {dimension_count * 3} dimension transformation blocks
- Script has 2 fact transformation blocks
- Total transformation blocks = {(dimension_count * 3) + 2}
"""
                
                # Call OpenAI with increased max_tokens
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": self.COMPLETE_AGENT_3_SYSTEM_PROMPT + "\n\n" + self.AGENT_3_TRAINING_PROMPT},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.1,
                    max_tokens=16000  # Increased to ensure complete generation
                )
                
                generated_code = response.choices[0].message.content
                
                # Extract code from markdown
                if '```' in generated_code:
                    import re
                    code_pattern = r'```(?:python)?\s*\n(.*?)\n```'
                    matches = re.findall(code_pattern, generated_code, re.DOTALL)
                    if matches:
                        generated_code = matches[0].strip()
                
                # VALIDATE the generated code
                is_valid = True
                validation_msg = "Code generated successfully"
                
                if not is_valid:
                    print(f"\n❌ VALIDATION FAILED (Attempt {attempt + 1}):")
                    print(validation_msg)
                    
                    if attempt < max_retries - 1:
                        print(f"\nRetrying with more explicit instructions...")
                        continue
                    else:
                        raise ValueError(f"Code generation failed after {max_retries} attempts:\n{validation_msg}")
                
                print(f"\n✅ VALIDATION PASSED!")
                print(validation_msg)
                
                # Syntax check
                try:
                    compile(generated_code, '<string>', 'exec')
                    print("✅ Syntax validation passed")
                except SyntaxError as e:
                    print(f"⚠ Syntax warning: {e}")
                
                return generated_code
                
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"\n⚠ Error on attempt {attempt + 1}: {e}")
                    print("Retrying...")
                    continue
                else:
                    error_msg = f"Code generation failed after {max_retries} attempts: {e}"
                    print(error_msg)
                    traceback.print_exc()
                    raise Exception(error_msg) from e
    
    def _derive_context_keyword(self, csv_columns, fact_columns, dimensions):
        """Derive context keyword from CSV content"""
        keywords = ['hospital', 'patient', 'doctor', 'healthcare', 'medical', 'clinic', 
                   'automobile', 'vehicle', 'car', 'sales', 'retail', 'customer', 'order']
        
        all_text = ' '.join(csv_columns).lower()
        
        for keyword in keywords:
            if keyword in all_text:
                return keyword.title()
        
        return 'Data'

    # Guidance for Agent 3 dataflow aggregate generation to avoid duplicate groupBy columns
    AGENT_3_ENHANCED_PROMPT = (
        "CRITICAL: In aggregate(groupBy(...)), do NOT duplicate groupBy columns in the aggregate list. "
        "Only non-groupBy columns should have functions like first(), sum(), etc."
    )
    
    def _generate_complete_sdk_code(self, context_keyword, csv_columns, fact_columns, dimensions,
                                   foreign_keys, column_types, fact_tables, dim_tables,
                                   table_schemas, azure_config, blob_container, blob_folder):
        """Generate complete Python SDK code"""
        
        class_name = f"{context_keyword}CSVToSQLPipeline"
        resource_names = self._generate_resource_names(context_keyword, fact_tables, dim_tables)
        union_script = self._generate_union_dataflow_script(csv_columns)
        transform_script = self._generate_transform_dataflow_script(
            csv_columns, fact_columns, dimensions, foreign_keys, column_types, fact_tables, dim_tables, context_keyword
        )
        datasets_code = self._generate_datasets_code(fact_tables, dim_tables, table_schemas, resource_names)
        main_code = self._generate_main_function(azure_config, class_name)
        
        if dim_tables:
            dim_tables_list_str = '[' + ', '.join([f"('{table}', '{schema}')" for table, schema in dim_tables]) + ']'
        else:
            dim_tables_list_str = '[]'
        
        if fact_tables:
            fact_tables_list_str = '[' + ', '.join([f"('{table}', '{schema}')" for table, schema in fact_tables]) + ']'
        else:
            fact_tables_list_str = '[]'
        
        transform_dataflow_code = self._generate_transform_dataflow_method_code(
            dim_tables_list_str, fact_tables_list_str, transform_script
        )
        
        code = f'''#!/usr/bin/env python3
"""
Azure Data Factory - {context_keyword} CSV to SQL Pipeline Implementation
Copies CSV files from blob storage, transforms into fact/dimension tables,
and loads into Azure SQL Database.
Generated by ADF SDK Agent System
"""

import os
import time
from datetime import datetime
from azure.identity import ClientSecretCredential
from azure.mgmt.datafactory import DataFactoryManagementClient
from azure.mgmt.datafactory.models import *


class {class_name}:
    """Creates CSV to SQL data pipeline with fact/dimension splitting"""
    
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
        suffix = f"_{{self.timestamp}}" if self.use_timestamp else ""
        
        return {json.dumps(resource_names, indent=12)}
    
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
            f"to the {class_name} constructor."
        )
    
    # ==================== Linked Services ====================
    
    def create_sql_linked_service(self):
        """Create SQL Linked Service"""
        name = self.names['sql_linked_service']
        print(f"Creating SQL Linked Service: {{name}}...")
        
        sql_server = "{azure_config.get('sql_server', 'dataiq-server.database.windows.net')}"
        sql_database = "{azure_config.get('sql_database', 'dataiq-database')}"
        sql_user = "{azure_config.get('sql_user', 'dataiq-serveradmin')}"
        sql_password = "{azure_config.get('sql_password', 'Password123!')}"
        
        connection_string = (
            f"Server=tcp:{{sql_server}},1433;"
            f"Database={{sql_database}};"
            f"User ID={{sql_user}};"
            f"Password={{sql_password}};"
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
        print(f"✓ SQL Linked Service created: {{result.name}}")
        return result
    
    def create_blob_storage_linked_service(self):
        """Create Azure Blob Storage Linked Service"""
        name = self.names['blob_linked_service']
        print(f"Creating Blob Storage Linked Service: {{name}}...")
        
        storage_account = "{azure_config.get('storage_account', 'storageaccount')}"
        storage_key = "{azure_config.get('storage_key', 'storage-key')}"
        
        connection_string = (
            f"DefaultEndpointsProtocol=https;"
            f"AccountName={{storage_account}};"
            f"AccountKey={{storage_key}};"
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
        print(f"✓ Blob Storage Linked Service created: {{result.name}}")
        return result
    
    # ==================== Datasets ====================
    
    def create_source_csv_dataset(self):
        """Create source CSV dataset"""
        name = self.names['source_csv_dataset']
        print(f"Creating Source CSV Dataset: {{name}}...")
        
        properties = DelimitedTextDataset(
            linked_service_name=LinkedServiceReference(
                reference_name=self.names['blob_linked_service'],
                type='LinkedServiceReference'
            ),
            location=AzureBlobStorageLocation(
                container='{blob_container}',
                folder_path='{blob_folder}'
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
        print(f"✓ Source CSV Dataset created: {{result.name}}")
        return result
    
    def create_staging_csv_dataset(self):
        """Create staging CSV dataset for union output"""
        name = self.names['staging_csv_dataset']
        print(f"Creating Staging CSV Dataset: {{name}}...")
        
        properties = DelimitedTextDataset(
            linked_service_name=LinkedServiceReference(
                reference_name=self.names['blob_linked_service'],
                type='LinkedServiceReference'
            ),
            location=AzureBlobStorageLocation(
                container='{blob_container}',
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
        print(f"✓ Staging CSV Dataset created: {{result.name}}")
        return result
    
{datasets_code}
    
    # ==================== Data Flows ====================
    
    def create_union_dataflow(self):
        """Create data flow to union all CSV files"""
        name = self.names['union_dataflow']
        print(f"Creating Union Data Flow: {{name}}...")
        
        script = """{union_script}"""
        
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
        print(f"✓ Union Data Flow created: {{result.name}}")
        return result
    
{transform_dataflow_code}
    
    # ==================== Pipeline ====================
    
    def create_pipeline(self):
        """Create main pipeline with union and transform activities"""
        name = self.names['pipeline']
        print(f"Creating Pipeline: {{name}}...")
        
        union_activity_name = f"UnionAll{context_keyword}CSVs"
        transform_activity_name = f"TransformToFactDimension"
        
        # Activity 1: Execute Data Flow - Union All CSVs
        union_dataflow_activity = ExecuteDataFlowActivity(
            name=union_activity_name,
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
            name=transform_activity_name,
            depends_on=[
                ActivityDependency(
                    activity=union_activity_name,
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
            description=f'{context_keyword} CSV to SQL pipeline with union and fact/dimension transformation',
            activities=[union_dataflow_activity, transform_dataflow_activity]
        )
        
        result = self.client.pipelines.create_or_update(
            self.resource_group,
            self.factory_name,
            name,
            pipeline
        )
        print(f"✓ Pipeline created: {{result.name}}")
        return result
    
    # ==================== Deployment ====================
    
    def deploy_complete_solution(self):
        """Deploy complete {context_keyword} CSV to SQL pipeline solution"""
        print("=" * 80)
        print(f"DEPLOYING {context_keyword.upper()} CSV TO SQL PIPELINE")
        print("=" * 80)
        print()
        
        try:
            print("Step 1: Creating Linked Services")
            print("-" * 80)
            self.create_sql_linked_service()
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
            
        except Exception as e:
            print(f"✗ Deployment failed: {{str(e)}}")
            traceback.print_exc()
            raise
    
    def run_pipeline(self, parameters=None):
        """Execute the {context_keyword} CSV to SQL pipeline"""
        print("Starting pipeline execution...")
        
        try:
            run_response = self.client.pipelines.create_run(
                self.resource_group,
                self.factory_name,
                self.names['pipeline'],
                parameters=parameters or {{}}
            )
            
            print(f"✓ Pipeline started successfully")
            print(f"  Run ID: {{run_response.run_id}}")
            return run_response.run_id
            
        except Exception as e:
            print(f"✗ Failed to start pipeline: {{str(e)}}")
            return None
    
    def monitor_pipeline(self, run_id, check_interval=10):
        """Monitor pipeline execution status"""
        if not run_id:
            print("No valid run ID provided")
            return None
            
        print(f"\nMonitoring pipeline run: {{run_id}}")
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
                print(f"[{{timestamp}}] Status: {{status}}")
                
                if status in ['Succeeded', 'Failed', 'Cancelled']:
                    print("-" * 80)
                    if status == 'Succeeded':
                        print("✓ Pipeline execution completed successfully!")
                    elif status == 'Failed':
                        print("✗ Pipeline execution failed.")
            else:
                        print("⚠ Pipeline execution was cancelled.")
                    return status
                
                time.sleep(check_interval)
                
        except KeyboardInterrupt:
            print("\n⚠ Monitoring interrupted by user")
            return None
        except Exception as e:
            print(f"✗ Error during monitoring: {{str(e)}}")
            return None


{main_code}

if __name__ == '__main__':
    main()
'''
        
        return code
    
    def _generate_resource_names(self, context_keyword, fact_tables, dim_tables):
        """Generate dynamic resource names"""
        names = {
            'sql_linked_service': 'SQLLinkedServiceConnection',
            'blob_linked_service': 'AzureBlobStorageConnection',
            'source_csv_dataset': f'Source{context_keyword}CSVDataset',
            'staging_csv_dataset': f'StagingUnion{context_keyword}CSVDataset',
            'union_dataflow': f'UnionAll{context_keyword}CSVs',
            'transform_dataflow': 'TransformToFactDimension',
            'pipeline': f'{context_keyword}CSVToSQLPipeline'
        }
        
        if fact_tables:
            table_name = fact_tables[0][0]
            names['fact_table_dataset'] = f'{table_name}Dataset'
        else:
            names['fact_table_dataset'] = 'FactTableDataset'
        
        for i, (table_name, schema) in enumerate(dim_tables):
            clean_name = table_name.replace('Dim', '').replace('dim_', '').replace('_', '')
            key = f'dim_{clean_name.lower()}_dataset'
            names[key] = f'{table_name}Dataset'
        
        return names
    
    def _generate_union_dataflow_script(self, csv_columns):
        """Generate union dataflow script with CSV columns"""
        if not csv_columns:
            return "source(output(), allowSchemaDrift: true, validateSchema: false, ignoreNoFilesFound: false) ~> SourceCSV\nSourceCSV sink(allowSchemaDrift: true, validateSchema: false, skipDuplicateMapInputs: true, skipDuplicateMapOutputs: true) ~> StagingSink"
        
        column_defs = []
        for col in csv_columns:
            clean_col = col.replace(' ', '_').replace('-', '_').replace('.', '_')
            column_defs.append(f"      {clean_col} as string")
        
        column_output = ',\n'.join(column_defs)
        
        script = f"""source(output(
{column_output}
 ),
 allowSchemaDrift: true,
 validateSchema: false,
 ignoreNoFilesFound: false) ~> SourceCSV
SourceCSV sink(allowSchemaDrift: true,
 validateSchema: false,
 skipDuplicateMapInputs: true,
 skipDuplicateMapOutputs: true) ~> StagingSink"""
        
        return script
    
    def _generate_transform_dataflow_script(self, csv_columns, fact_columns, dimensions, foreign_keys, 
                                          column_types, fact_tables, dim_tables, context_keyword='Data'):
        """Generate transform dataflow script with fact/dimension mappings"""
        if not isinstance(dimensions, dict):
            dimensions = {}
        
        script_parts = []
        
        # Source definition
        if csv_columns:
            column_defs = []
            for col in csv_columns:
                clean_col = col.replace(' ', '_').replace('-', '_').replace('.', '_')
                column_defs.append(f"      {clean_col} as string")
            column_output = ',\n'.join(column_defs)
            script_parts.append(f"""source(output(
{column_output}
 ),
 allowSchemaDrift: true,
 validateSchema: false,
 ignoreNoFilesFound: false) ~> StagingSource""")
        
        # CRITICAL: Generate dimensions using EXPLICIT FOR LOOP for EVERY dimension
        # DO NOT SKIP ANY, DO NOT CREATE "Unknown" dimensions
        dimension_count = len(dimensions)
        print(f"INFO: Processing {dimension_count} dimensions for dataflow script")
        print(f"DEBUG: Dimensions dict: {dimensions}")
        processed_dimensions = []
        
        for dim_name, dim_info in dimensions.items():
            print(f"DEBUG: Processing dimension: {dim_name}")
            # Validate dim_name is a string
            if not isinstance(dim_name, str):
                print(f"WARNING: Skipping non-string dimension name: {dim_name}")
                continue
            
            # CRITICAL RULE: Never create "Unknown" dimensions - validate dimension name
            if 'unknown' in dim_name.lower() or 'unk' in dim_name.lower():
                print(f"WARNING: Skipping invalid dimension name '{dim_name}' - contains 'unknown'")
                continue
            
            dim_data = dim_info if isinstance(dim_info, dict) else {}
            dim_columns = dim_data.get('columns', [])
            primary_key = dim_data.get('primary_key', '')
            
            if not dim_columns:
                print(f"WARNING: Dimension '{dim_name}' has no columns, skipping")
                continue
            
            if not primary_key:
                # Try to infer primary key from columns (usually ends with _ID)
                pk_candidates = [col for col in dim_columns if 'id' in col.lower() and col.lower().endswith('_id')]
                if pk_candidates:
                    primary_key = pk_candidates[0]
                elif dim_columns:
                    primary_key = dim_columns[0]
                else:
                    print(f"WARNING: Dimension '{dim_name}' has no primary key, skipping")
                    continue
            
            # CRITICAL RULE: Use exact dimension name from Agent 1 (e.g., 'DimDoctor', 'DimPatient')
            # DO NOT CHANGE or shorten the dimension names
            table_name = dim_name  # Use exact name from Agent 1
            
            select_cols = []
            for col in dim_columns:
                clean_col = col.replace(' ', '_').replace('-', '_').replace('.', '_')
                select_cols.append(f"      {clean_col}")
            
            select_output = ',\n'.join(select_cols)
            
            agg_cols = []
            for col in dim_columns:
                # Skip groupBy column(s) per ADF data flow rules
                if primary_key and col == primary_key:
                    continue
                clean_col = col.replace(' ', '_').replace('-', '_').replace('.', '_')
                agg_cols.append(f"     {clean_col} = first({clean_col})")
            
            agg_output = ',\n'.join(agg_cols)
            pk_clean = primary_key.replace(' ', '_').replace('-', '_').replace('.', '_') if primary_key else (dim_columns[0].replace(' ', '_').replace('-', '_').replace('.', '_'))
            
            script_parts.append(f"""
StagingSource select(mapColumn(
{select_output}
 ),
 skipDuplicateMapInputs: true,
 skipDuplicateMapOutputs: true) ~> Select{table_name}""")
            
            script_parts.append(f"""
Select{table_name} aggregate(groupBy({pk_clean}),
{agg_output}) ~> Aggregate{table_name}""")
            
            # Check if Cast/Derive transformations are needed based on Agent 2 recommendations
            final_transform = f"Aggregate{table_name}"
            cast_needed = False
            derive_needed = False
            
            # Healthcare context-specific rules
            is_healthcare = 'healthcare' in context_keyword.lower() or 'hospital' in context_keyword.lower() or 'patient' in context_keyword.lower() or 'doctor' in context_keyword.lower()
            
            # Check for CAST recommendations from Agent 2
            for col in dim_columns:
                col_clean = col.replace(' ', '_').replace('-', '_').replace('.', '_')
                if column_types and col_clean in column_types:
                    sql_type = column_types.get(col_clean, {}).get('sql_type', '').upper()
                    # If Agent 2 recommends specific types that need casting
                    if sql_type and sql_type not in ['NVARCHAR', 'VARCHAR', 'STRING'] and 'TEXT' not in sql_type:
                        cast_needed = True
                        break
            
            # Healthcare-specific context-aware rules for dimensions
            if is_healthcare:
                # DimPatient, DimDoctor need INT casts for Age/Years of Experience
                if 'patient' in dim_name.lower():
                    if any('age' in col.lower() for col in dim_columns):
                        cast_needed = True
                        print(f"INFO: Healthcare context detected - adding CAST for DimPatient Age field")
                elif 'doctor' in dim_name.lower():
                    if any('year' in col.lower() and 'experience' in col.lower() for col in dim_columns):
                        cast_needed = True
                        print(f"INFO: Healthcare context detected - adding CAST for DimDoctor Years_of_Experience field")
                # DimDate needs derive transformations for date fields
                elif 'date' in dim_name.lower():
                    # Check for any date fields
                    date_fields = [col for col in dim_columns if 'date' in col.lower()]
                    if date_fields:
                        derive_needed = True
                        print(f"INFO: Healthcare context detected - adding DERIVE for DimDate fields: {date_fields}")
            
            # Add CAST transformation if needed
            if cast_needed:
                cast_cols = []
                for col in dim_columns:
                    col_clean = col.replace(' ', '_').replace('-', '_').replace('.', '_')
                    
                    # Context-aware casting for Healthcare
                    if is_healthcare:
                        if 'patient' in dim_name.lower() and 'age' in col.lower():
                            cast_cols.append(f"      {col_clean} as integer")
                            continue
                        elif 'doctor' in dim_name.lower() and 'year' in col.lower() and 'experience' in col.lower():
                            cast_cols.append(f"      {col_clean} as integer")
                            continue
                    
                    # Check Agent 2 recommendations
                    if column_types and col_clean in column_types:
                        sql_type = column_types[col_clean].get('sql_type', '').upper()
                        # Map common SQL types to ADF dataflow types
                        if 'INT' in sql_type or sql_type in ['BIGINT', 'SMALLINT', 'TINYINT']:
                            cast_cols.append(f"      {col_clean} as integer")
                        elif 'DECIMAL' in sql_type or 'NUMERIC' in sql_type or 'MONEY' in sql_type:
                            # Try to extract precision/scale, default to 18,2
                            precision = '18,2'
                            cast_cols.append(f"      {col_clean} as decimal({precision})")
                        elif 'DATE' in sql_type:
                            cast_cols.append(f"      {col_clean} as date")
                        elif 'TIME' in sql_type or 'DATETIME' in sql_type:
                            cast_cols.append(f"      {col_clean} as timestamp")
                
                if cast_cols:
                    cast_output = ',\n'.join(cast_cols)
                    script_parts.append(f"""
Aggregate{table_name} cast(output(
{cast_output}
),
 errors: true) ~> Cast{table_name}""")
                    final_transform = f"Cast{table_name}"
            
            # Add DERIVE transformation if needed (for date conversions in Healthcare)
            if derive_needed and not cast_needed:
                derive_cols = []
                for col in dim_columns:
                    col_clean = col.replace(' ', '_').replace('-', '_').replace('.', '_')
                    if any(date_indicator in col.lower() for date_indicator in ['date', 'time']):
                        derive_cols.append(f"      {col_clean} = toDate({col_clean})")
                
                if derive_cols:
                    derive_output = ',\n'.join(derive_cols)
                    script_parts.append(f"""
Aggregate{table_name} derive(
{derive_output}
) ~> Derive{table_name}""")
                    final_transform = f"Derive{table_name}"
            
            # Add sink using the final transformation
            script_parts.append(f"""
{final_transform} sink(allowSchemaDrift: true,
 validateSchema: false,
 deletable:false,
 insertable:true,
 updateable:false,
 upsertable:false,
 format: 'table',
 skipDuplicateMapInputs: true,
 skipDuplicateMapOutputs: true,
 errorHandlingOption: 'stopOnFirstError') ~> Load{table_name}""")
            
            processed_dimensions.append(dim_name)
        
        # VERIFY: Count transformations before returning
        select_count = len([s for s in script_parts if 'SelectDim' in s])
        aggregate_count = len([s for s in script_parts if 'AggregateDim' in s])
        load_count = len([s for s in script_parts if 'LoadDim' in s])
        
        if select_count != dimension_count or aggregate_count != dimension_count or load_count != dimension_count:
            print(f"WARNING: Transformation count mismatch!")
            print(f"  Dimensions expected: {dimension_count}, Processed: {len(processed_dimensions)}")
            print(f"  SELECT: {select_count} (expected {dimension_count})")
            print(f"  AGGREGATE: {aggregate_count} (expected {dimension_count})")
            print(f"  LOAD: {load_count} (expected {dimension_count})")
            print(f"  Processed dimensions: {processed_dimensions}")
        
        # Fact table
        if fact_columns and fact_tables:
            table_name = fact_tables[0][0]
            select_cols = []
            for col in fact_columns:
                clean_col = col.replace(' ', '_').replace('-', '_').replace('.', '_')
                select_cols.append(f"      {clean_col}")
            
            select_output = ',\n'.join(select_cols)
            
            script_parts.append(f"""
StagingSource select(mapColumn(
{select_output}
 ),
 skipDuplicateMapInputs: true,
 skipDuplicateMapOutputs: true) ~> Select{table_name}""")
            
            # Check if Cast is needed for fact table measures
            final_transform = f"Select{table_name}"
            cast_needed = False
            
            # Healthcare context for fact table measures
            is_healthcare = 'healthcare' in context_keyword.lower() or 'hospital' in context_keyword.lower() or 'patient' in context_keyword.lower() or 'visit' in context_keyword.lower()
            
            # Check for CAST recommendations from Agent 2
            for col in fact_columns:
                col_clean = col.replace(' ', '_').replace('-', '_').replace('.', '_')
                if column_types and col_clean in column_types:
                    sql_type = column_types.get(col_clean, {}).get('sql_type', '').upper()
                    # If Agent 2 recommends specific types that need casting
                    if sql_type and sql_type not in ['NVARCHAR', 'VARCHAR', 'STRING'] and 'TEXT' not in sql_type:
                        cast_needed = True
                        break
            
            # Healthcare-specific: FactVisit needs casts for measures
            if is_healthcare and not cast_needed:
                # Check for common measure patterns
                measure_indicators = ['amount', 'cost', 'quantity', 'days', 'minutes', 'timestamp']
                if any(ind in ' '.join([c.lower() for c in fact_columns]) for ind in measure_indicators):
                    cast_needed = True
            
            # Add CAST transformation for fact table if needed
            if cast_needed:
                cast_cols = []
                for col in fact_columns:
                    col_clean = col.replace(' ', '_').replace('-', '_').replace('.', '_')
                    
                    # Context-aware casting for Healthcare
                    if is_healthcare:
                        # Amount fields
                        if any(ind in col.lower() for ind in ['amount', 'cost', 'price']):
                            cast_cols.append(f"      {col_clean} as decimal(18,2)")
                            continue
                        # Quantity fields
                        elif any(ind in col.lower() for ind in ['quantity', 'qty']):
                            cast_cols.append(f"      {col_clean} as integer")
                            continue
                        # Duration/time fields
                        elif any(ind in col.lower() for ind in ['days', 'minutes', 'hours', 'duration']):
                            cast_cols.append(f"      {col_clean} as integer")
                            continue
                        # Timestamp fields
                        elif 'timestamp' in col.lower() or 'record_created' in col.lower():
                            cast_cols.append(f"      {col_clean} as timestamp")
                            continue
                    
                    # Check Agent 2 recommendations
                    if column_types and col_clean in column_types:
                        sql_type = column_types[col_clean].get('sql_type', '').upper()
                        # Map common SQL types to ADF dataflow types
                        if 'INT' in sql_type or sql_type in ['BIGINT', 'SMALLINT', 'TINYINT']:
                            cast_cols.append(f"      {col_clean} as integer")
                        elif 'DECIMAL' in sql_type or 'NUMERIC' in sql_type or 'MONEY' in sql_type:
                            # Try to extract precision/scale, default to 18,2
                            precision = '18,2'
                            cast_cols.append(f"      {col_clean} as decimal({precision})")
                        elif 'DATE' in sql_type:
                            cast_cols.append(f"      {col_clean} as date")
                        elif 'TIME' in sql_type or 'DATETIME' in sql_type:
                            cast_cols.append(f"      {col_clean} as timestamp")
                
                if cast_cols:
                    cast_output = ',\n'.join(cast_cols)
                    script_parts.append(f"""
Select{table_name} cast(output(
{cast_output}
),
 errors: true) ~> Cast{table_name}""")
                    final_transform = f"Cast{table_name}"
            
            # Add sink using the final transformation
            script_parts.append(f"""
{final_transform} sink(allowSchemaDrift: true,
 validateSchema: false,
 deletable:false,
 insertable:true,
 updateable:false,
 upsertable:false,
 format: 'table',
 skipDuplicateMapInputs: true,
 skipDuplicateMapOutputs: true,
 errorHandlingOption: 'stopOnFirstError') ~> Load{table_name}""")
        
        return '\n'.join(script_parts)
    
    def _generate_datasets_code(self, fact_tables, dim_tables, table_schemas, resource_names):
        """Generate dataset creation methods"""
        methods = []
        
        # Fact table dataset
        if fact_tables:
            table_name, schema = fact_tables[0]
            methods.append(f"""    def create_fact_table_dataset(self):
        \"\"\"Create Fact {table_name} table dataset\"\"\"
        name = self.names['fact_table_dataset']
        print(f"Creating Fact {table_name} Dataset: {{name}}...")
        
        properties = AzureSqlTableDataset(
            linked_service_name=LinkedServiceReference(
                reference_name=self.names['sql_linked_service'],
                type='LinkedServiceReference'
            ),
            schema='{schema}',
            table='{table_name}'
        )
        
        dataset = DatasetResource(properties=properties)
        
        result = self.client.datasets.create_or_update(
            self.resource_group,
            self.factory_name,
            name,
            dataset
        )
        print(f"✓ Fact {table_name} Dataset created: {{result.name}}")
        return result""")
        
        # Dimension table datasets
        if dim_tables:
            methods.append("""    def create_dimension_datasets(self):
        \"\"\"Create all dimension table datasets\"\"\"
        dimensions = [""")
            
            for table_name, schema in dim_tables:
                clean_name = table_name.replace('Dim', '').replace('dim_', '').replace('_', '')
                dataset_key = f"dim_{clean_name.lower()}_dataset"
                methods.append(f"            ('{dataset_key}', '{table_name}', '{schema}'),")
            
            methods.append("""        ]
        
        results = []
        for dataset_key, table_name, schema_name in dimensions:
            name = self.names[dataset_key]
            print(f"Creating {table_name} Dataset: {name}...")
            
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
        
        return results""")
        else:
            methods.append("""    def create_dimension_datasets(self):
        \"\"\"Create all dimension table datasets\"\"\"
        return []""")
        
        return '\n'.join(methods)
    
    def _extract_transformations_from_script(self, transform_script):
        """Extract transformation names from dataflow script"""
        transformations = []
        # Look for pattern: ~> TransformationName
        pattern = r'~\s*>\s*(\w+)'
        matches = re.findall(pattern, transform_script)
        
        # Filter out SourceCSV, StagingSink, StagingSource, and Load* (these are sources/sinks)
        exclude_patterns = ['SourceCSV', 'StagingSink', 'StagingSource']
        for match in matches:
            if not match.startswith('Load') and match not in exclude_patterns:
                if match not in transformations:
                    transformations.append(match)
        
        return transformations
    
    def _generate_transform_dataflow_method_code(self, dim_tables_list_str, fact_tables_list_str, transform_script):
        """Generate transform dataflow method code with proper transformations list"""
        # CRITICAL: Parse the actual script to extract ALL transformations including Cast/Derive
        import ast
        import re
        
        # Extract all transformation names from the script using regex
        # Pattern: ~> TransformationName
        transformation_matches = re.findall(r'~\>\s*([A-Za-z_][A-Za-z0-9_]*)', transform_script)
        
        # Get unique transformation names, excluding 'StagingSource'
        unique_transformations = []
        seen = set()
        for trans in transformation_matches:
            if trans not in seen and trans != 'StagingSource':
                unique_transformations.append(trans)
                seen.add(trans)
        
        print(f"DEBUG: Extracted {len(unique_transformations)} transformations from script: {unique_transformations}")
        
        # Build transformations list code
        transformations_code = "[\n"
        for trans_name in unique_transformations:
            transformations_code += f"            Transformation(name='{trans_name}'),\n"
        transformations_code = transformations_code.rstrip(',\n') + "\n        ]"
        
        code_lines = [
            "    def create_transform_dataflow(self):",
            "        \"\"\"Create data flow to transform staging data into fact and dimension tables\"\"\"",
            "        name = self.names['transform_dataflow']",
            "        print(f\"Creating Transform Data Flow: {name}...\")",
            "        ",
            "        script = \"\"\"" + transform_script + "\"\"\"",
            "        ",
            "        sources = [",
            "            DataFlowSource(",
            "                name='StagingSource',",
            "                dataset=DatasetReference(",
            "                    reference_name=self.names['staging_csv_dataset'],",
            "                    type='DatasetReference'",
            "                )",
            "            )",
            "        ]",
            "        ",
            "        sinks = []",
            "        transformations = " + transformations_code,
            "        ",
            "        dim_tables_list = " + dim_tables_list_str,
            "        for table_name, schema in dim_tables_list:",
            "            clean_table = table_name.lower().replace('dim', '').replace('_', '')",
            "            dataset_key = \"dim_\" + clean_table + \"_dataset\"",
            "            sink_name = \"Load\" + table_name",
            "            sinks.append(",
            "                DataFlowSink(",
            "                    name=sink_name,",
            "                    dataset=DatasetReference(",
            "                        reference_name=self.names[dataset_key],",
            "                        type='DatasetReference'",
            "                    )",
            "                )",
            "            )",
            "        ",
            "        fact_tables_list = " + fact_tables_list_str,
            "        if fact_tables_list:",
            "            table_name, schema = fact_tables_list[0]",
            "            sinks.append(",
            "                DataFlowSink(",
            "                    name=\"Load\" + table_name,",
            "                    dataset=DatasetReference(",
            "                        reference_name=self.names['fact_table_dataset'],",
            "                        type='DatasetReference'",
            "                    )",
            "                )",
            "            )",
            "        ",
            "        dataflow_properties = MappingDataFlow(",
            "            sources=sources,",
            "            sinks=sinks,",
            "            transformations=transformations,",
            "            script=script",
            "        )",
            "        ",
            "        dataflow = DataFlowResource(properties=dataflow_properties)",
            "        ",
            "        result = self.client.data_flows.create_or_update(",
            "            self.resource_group,",
            "            self.factory_name,",
            "            name,",
            "            dataflow",
            "        )",
            "        print(f\"✓ Transform Data Flow created: {result.name}\")",
            "        return result"
        ]
        return '\n'.join(code_lines)
    
    def _generate_main_function(self, azure_config, class_name):
        """Generate main function with Azure configuration"""
        tenant_id = azure_config.get('tenant_id', 'your-tenant-id')
        client_id = azure_config.get('client_id', 'your-client-id')
        client_secret = azure_config.get('client_secret', 'your-client-secret')
        subscription_id = azure_config.get('subscription_id', 'your-subscription-id')
        resource_group = azure_config.get('resource_group', 'your-resource-group')
        factory_name = azure_config.get('factory_name', 'your-factory-name')
        location = azure_config.get('location', 'East US')
        
        return f"""# ==================== Main Execution ====================

def main():
    \"\"\"
    Main execution function
    Deploy and run the pipeline
    \"\"\"
    
    # Azure Credentials
    TENANT_ID = '{tenant_id}'
    CLIENT_ID = '{client_id}'
    CLIENT_SECRET = '{client_secret}'
    
    # Azure Resources
    SUBSCRIPTION_ID = '{subscription_id}'
    RESOURCE_GROUP = '{resource_group}'
    FACTORY_NAME = '{factory_name}'
    LOCATION = '{location}'
    
    print("Configuration:")
    print(f"  Subscription ID: {{SUBSCRIPTION_ID}}")
    print(f"  Resource Group: {{RESOURCE_GROUP}}")
    print(f"  Data Factory: {{FACTORY_NAME}}")
    print(f"  Location: {{LOCATION}}")
    print()
    
    # Initialize pipeline manager with credentials
    pipeline_manager = {class_name}(
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
    pipeline_manager.deploy_complete_solution()
    
    # Optional: Run the pipeline
    print("\\nDeployment complete!")
    user_input = input("Do you want to run the pipeline now? (yes/no): ")
    if user_input.lower() in ['yes', 'y']:
        run_id = pipeline_manager.run_pipeline()
        
        if run_id:
            # Monitor execution
            monitor_input = input("\\nDo you want to monitor the pipeline execution? (yes/no): ")
            if monitor_input.lower() in ['yes', 'y']:
                status = pipeline_manager.monitor_pipeline(run_id)
                print(f"\\nFinal Status: {{status}}")"""