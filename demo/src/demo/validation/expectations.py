import great_expectations as ge
from great_expectations.core.batch import RuntimeBatchRequest
from great_expectations.core.expectation_configuration import ExpectationConfiguration
from typing import Dict, Any, List
import pandas as pd


class DataQualityValidator:
    """Data quality validation using Great Expectations"""
    
    def __init__(self):
        self.context = ge.get_context()
        self.expectations = self._define_expectations()
    
    def _define_expectations(self) -> List[ExpectationConfiguration]:
        """Define data quality expectations"""
        return [
            ExpectationConfiguration(
                expectation_type="expect_table_row_count_to_be_between",
                kwargs={"min_value": 100}
            ),
            ExpectationConfiguration(
                expectation_type="expect_column_values_to_not_be_null",
                kwargs={"column": "sample_id"}
            ),
            ExpectationConfiguration(
                expectation_type="expect_column_values_to_be_unique",
                kwargs={"column": "sample_id"}
            ),
            ExpectationConfiguration(
                expectation_type="expect_column_values_to_be_between",
                kwargs={
                    "column": "label",
                    "min_value": 0,
                    "max_value": 1,
                    "mostly": 0.99
                }
            ),
        ]
    
    def validate_dataset(self, df: pd.DataFrame, dataset_name: str) -> Dict[str, Any]:
        """Validate dataset against expectations"""
        
        # Create a batch from the dataframe
        batch_request = RuntimeBatchRequest(
            datasource_name="pandas_datasource",
            data_connector_name="runtime_data_connector",
            data_asset_name=dataset_name,
            runtime_parameters={"batch_data": df},
            batch_identifiers={"default_identifier_name": "default_identifier"},
        )
        
        # Add or update expectation suite
        suite_name = f"{dataset_name}_expectations"
        suite = self.context.create_expectation_suite(
            expectation_suite_name=suite_name,
            overwrite_existing=True
        )
        
        # Add expectations to suite
        for expectation in self.expectations:
            suite.add_expectation(expectation_configuration=expectation)
        
        # Run validation
        validator = self.context.get_validator(
            batch_request=batch_request,
            expectation_suite_name=suite_name
        )
        
        results = validator.validate()
        
        return {
            "success": results.success,
            "statistics": results.statistics,
            "results": results.results
        }
