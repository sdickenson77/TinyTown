import os

import pytest
from pyspark.sql import SparkSession
from pyspark.sql.types import *
from tiny_town import (
    create_spark_session,
    fee_table,
    get_schema_definitions,
    validate_speeding_tickets,
    process_people_data
)



@pytest.fixture(scope="session")
def spark():
    # Set Java Home environment variable if not already set
    os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-11-openjdk-amd64"

    """Fixture for creating a Spark Session"""
    spark = (SparkSession.builder
            .master("local[10]")
            .appName("TinyTownTests")
            .getOrCreate())
    yield spark
    spark.stop()

@pytest.mark.parametrize("school_zone,construction,expected_fee", [
    (True, True, 120),
    (True, False, 60),
    (False, True, 60),
    (False, False, 30),
    (None, None, 0)
])
def test_fee_table(school_zone, construction, expected_fee):
    """Test fee calculation with different combinations"""
    assert fee_table(school_zone, construction) == expected_fee

def test_get_schema_definitions():
    """Test schema definition structure"""
    schema = get_schema_definitions()
    
    assert isinstance(schema, StructType)
    speeding_tickets_field = schema.fields[0]
    assert speeding_tickets_field.name == "speeding_tickets"
    assert isinstance(speeding_tickets_field.dataType, ArrayType)
    
    ticket_struct = speeding_tickets_field.dataType.elementType
    expected_fields = {
        "id": StringType,
        "license_plate": StringType,
        "officer_id": StringType,
        "recorded_mph_over_limit": LongType,
        "school_zone_ind": BooleanType,
        "speed_limit": LongType,
        "ticket_time": StringType,
        "work_zone_ind": BooleanType
    }
    
    for field_name, field_type in expected_fields.items():
        field = [f for f in ticket_struct.fields if f.name == field_name][0]
        assert isinstance(field.dataType, field_type)

def test_validate_speeding_tickets(spark):
    """Test ticket validation logic"""
    # Create sample test data
    test_data = [
        # Valid record
        ("T001", "ABC123", "OFF1", 15, True, 30, "2023-01-01", True),
        # Invalid record - missing license plate
        ("T002", None, "OFF1", 20, True, 30, "2023-01-01", True),
        # Invalid record - future date
        ("T003", "ABC124", "OFF1", 25, True, 30, "2026-01-01", True),
    ]
    
    columns = ["id", "license_plate", "officer_id", "recorded_mph_over_limit",
               "school_zone_ind", "speed_limit", "ticket_time", "work_zone_ind"]
    
    test_df = spark.createDataFrame(test_data, columns)
    validated_df = validate_speeding_tickets(test_df)
    
    # Check validation results
    assert validated_df.filter("is_valid = true").count() == 1
    assert validated_df.filter("is_valid = false").count() == 2
    
    # Check specific validation errors
    missing_plate = validated_df.filter("id = 'T002'").first()
    assert "Missing license plate" in missing_plate["validation_errors"]
    
    future_date = validated_df.filter("id = 'T003'").first()
    assert "Invalid future date" in future_date["validation_errors"]



# Additional helper fixtures for testing
@pytest.fixture
def sample_tickets_df(spark):
    """Fixture providing sample tickets data"""
    data = [
        ("T001", "ABC123", "OFF1", 15, True, 30, "2023-01-01", True),
        ("T002", "XYZ789", "OFF2", 20, False, 45, "2023-01-02", False)
    ]
    columns = ["id", "license_plate", "officer_id", "recorded_mph_over_limit",
               "school_zone_ind", "speed_limit", "ticket_time", "work_zone_ind"]
    return spark.createDataFrame(data, columns)

