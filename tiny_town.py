import os
from pyspark.sql import SparkSession, Window
from pyspark.sql.types import *
from pyspark.sql.functions import *
from datetime import datetime
from pyspark.sql.functions import col, when, to_timestamp, year, current_timestamp


def create_spark_session():
    """Create and configure Spark session."""
    # Set Java Home environment variable if not already set
    os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-11-openjdk-amd64"
    
    spark = SparkSession.builder \
        .appName("TinyTown Data Processing") \
        .config("spark.driver.extraJavaOptions", "-XX:+UseG1GC") \
        .config("spark.executor.extraJavaOptions", "-XX:+UseG1GC") \
        .config("spark.driver.memory", "4g") \
        .config("spark.executor.memory", "4g") \
        .config("spark.jars.packages", "com.databricks:spark-xml_2.12:0.15.0") \
        .config("spark.driver.extraJavaOptions", "-Dlog4j.configuration=log4j.properties") \
        .config("spark.python.worker.reuse", "true") \
        .getOrCreate()

    return spark


def fee_table(school_zone, construction):
    """Calculate fee based on zone conditions."""
    match school_zone, construction:
        case True, True:
            return 120
        case True, False:
            return 60
        case False, True:
            return 60
        case False, False:
            return 30
        case _:
            return 0


def process_people_data(spark):
    """Process and return people data DataFrame."""
    initial_people_df = spark.read \
        .option("mode", "permissive") \
        .option("header", True) \
        .option("inferSchema", True) \
        .option("delimiter", "|") \
        .csv("ttpd_data/*_people_*.csv")

    new_people_df = initial_people_df.toDF(*[c.replace('"', '').strip() for c in initial_people_df.columns])
    
    print("\nUnique fields in the dataset:")
    new_people_df.printSchema()
    print(f"\nTotal number of records: {new_people_df.count()}")
    print("\nSample of the final dataset:")
    new_people_df.show(5, False)
    
    return new_people_df


def get_schema_definitions():
    """Define and return schema structures."""
    ticket_schema = StructType([
        StructField("speeding_tickets", ArrayType(
            StructType([
                StructField("id", StringType(), True),
                StructField("license_plate", StringType(), True),
                StructField("officer_id", StringType(), True),
                StructField("recorded_mph_over_limit", LongType(), True),
                StructField("school_zone_ind", BooleanType(), True),
                StructField("speed_limit", LongType(), True),
                StructField("ticket_time", StringType(), True),
                StructField("work_zone_ind", BooleanType(), True)
            ])
        ), True)
    ])

    return ticket_schema


def load_data(spark, ticket_schema):
    """Load data from various sources."""
    # Load tickets data
    tickets_df = spark.read \
        .option("multiline", "true") \
        .schema(ticket_schema) \
        .json("ttpd_data/*speeding_tickets_*.json")
    
    tickets_df = tickets_df.select(explode("speeding_tickets").alias("ticket")) \
        .select("ticket.*")
    print("\nTicket Data:")
    tickets_df.show(5, False)

    # Load people data
    people_df = process_people_data(spark)

    # Load automobiles data
    automobiles_df = spark.read \
        .format("com.databricks.spark.xml") \
        .option("rowTag", "automobile") \
        .load("ttpd_data/*_automobiles_*.xml")

    return tickets_df, people_df, automobiles_df


def analyze_officer_tickets(tickets_df, people_df):
    """Analyze tickets issued by officers."""
    officer_tickets = tickets_df.join(people_df, tickets_df.officer_id == people_df.id) \
        .where(people_df.profession == "Police Officer") \
        .groupBy("officer_id", "first_name", "last_name") \
        .count() \
        .orderBy("count", ascending=False)

    print("\nQuestion 1: Which police officer was handed the most speeding tickets?")
    officer_tickets.show(1, False)


def analyze_ticket_trends(tickets_df):
    """Analyze ticket trends by different time periods."""
    # Monthly counts
    monthly_counts = tickets_df \
        .withColumn("year", year("ticket_time")) \
        .withColumn("month", month("ticket_time")) \
        .withColumn("year_month", concat(year("ticket_time"), month("ticket_time"))) \
        .groupBy("year_month", "year", "month") \
        .agg(count("*").alias("ticket_count")) \
        .orderBy("ticket_count", ascending=False)

    # Yearly trend
    yearly_trend = tickets_df \
        .withColumn("year", year("ticket_time")) \
        .groupBy("year") \
        .agg(count("*").alias("ticket_count")) \
        .orderBy("year")

    # Monthly trend
    monthly_trend = tickets_df \
        .withColumn("month", date_format("ticket_time", "MMM")) \
        .withColumn("month_num", month("ticket_time")) \
        .groupBy("month", "month_num") \
        .agg(count("*").alias("ticket_count")) \
        .orderBy("month_num")

    # Monthly-yearly trend
    monthly_yearly_trend = tickets_df \
        .withColumn("year", year("ticket_time")) \
        .withColumn("month", month("ticket_time")) \
        .groupBy("year", "month") \
        .agg(count("*").alias("ticket_count")) \
        .orderBy("year", "month")

    print("\n Question 2: What 3 months(year + month) had the most speeding tickets?")
    print("\nTop 3 months with most speeding tickets:")
    monthly_counts.show(3)

    print("\n Bonus: What overall month-by-month or year by year trends, if any, do you see?")
    print("\n Overall it seems that the most tickets are issued in December, followed by a noticeable increase in the summer months.")
    print("\n Yearly it seems there was a big jump from 2020 to 2021.")

    print("Yearly Trend:")
    yearly_trend.show()
    print("Monthly Trend:")
    monthly_trend.show()
    print("Monthly-Yearly Trend:")
    monthly_yearly_trend.show(50)


def analyze_ticket_costs(spark, tickets_df, automobiles_df, people_df):
    """Analyze ticket costs and fees."""
    fee_calculator = udf(lambda school_zone, work_zone:
                        fee_table(school_zone, work_zone), IntegerType())

    ticket_costs = tickets_df \
        .join(automobiles_df, tickets_df.license_plate == automobiles_df.license_plate) \
        .join(people_df, automobiles_df.person_id == people_df.id) \
        .withColumn("fee", fee_calculator(col("school_zone_ind"), col("work_zone_ind"))) \
        .groupBy("person_id", "first_name", "last_name") \
        .agg(
            sum("fee").alias("total_fees_paid"),
            count("*").alias("number_of_tickets")
        ) \
        .orderBy(col("total_fees_paid").desc())

    print("\n Question 3: Who are the top 10 people who have spent the most money paying speeding tickets overall?")
    print("\nTop 10 People Who Paid Most in Speeding Tickets:")
    ticket_costs.show(10, False)


def validate_speeding_tickets(tickets_df):
    """
    Validate speeding ticket data for accuracy and data quality.
    Returns a DataFrame with validation flags and a summary of validation results.
    """
    # Create a DataFrame with validation checks
    validated_df = tickets_df.withColumn(
        "validation_errors", 
        concat_ws(", ", 
            # Basic null checks for required fields
      when(col("id").isNull(), "Missing ticket ID").otherwise(""),
            when(col("license_plate").isNull(), "Missing license plate").otherwise(""),
            when(col("officer_id").isNull(), "Missing officer ID").otherwise(""),

            #speed validation
            when(col("speed_limit").isNull(), "Missing speed limit").otherwise(""),
            
            # Time validation
            when(col("ticket_time").isNull(), "Missing timestamp").otherwise(""),
            when(year(to_timestamp(col("ticket_time"))) < 2020, "Invalid year (before 2020)").otherwise(""),
            when(year(to_timestamp(col("ticket_time"))) > year(current_timestamp()), "Invalid future date").otherwise(""),
            
            # Boolean field validation
            when(col("school_zone_ind").isNull(), "Missing school zone indicator").otherwise(""),
            when(col("work_zone_ind").isNull(), "Missing work zone indicator").otherwise("")
        )
    )
    
    # Add a flag for records with validation errors
    validated_df = validated_df.withColumn(
        "is_valid",
        (~col("validation_errors").like("%Missing%")  & ~col("validation_errors").like("%Invalid%"))
    )
    
    # Generate validation summary
    total_records = tickets_df.count()
    invalid_records = validated_df.filter(col("is_valid") == False).count()

    validated_df.show(5, False)
    validated_df.printSchema()


    print("\n=== Validation Summary ===")
    print(f"Total records processed: {total_records}")
    print(f"Valid records: {total_records - invalid_records}")
    print(f"Invalid records: {invalid_records}")
    print(f"Data quality score: {((total_records - invalid_records) / total_records * 100):.2f}%")
    
    # Show detailed error breakdown
    print("\n=== Error Type Breakdown ===")
    error_breakdown = validated_df.filter(col("is_valid") == False) \
        .select(explode(split(col("validation_errors"), ",")).alias("error_type")) \
        .filter(col("error_type") != "") \
        .groupBy("error_type") \
        .count() \
        .orderBy(col("count").desc())
    
    error_breakdown.show(truncate=False)
    
    # Show sample invalid records
    print("\n=== Sample Invalid Records ===")
    validated_df.filter(col("is_valid") == False) \
        .select("id", "license_plate", "recorded_mph_over_limit", 
                "speed_limit", "ticket_time", "validation_errors") \
        .show(5, truncate=False)
    
    return validated_df

def check_ticket_relationships(tickets_df, people_df, automobiles_df):
    """
    Validate relationships between tickets and related entities.
    """
    print("\n=== Relationship Validation ===")
    
    # Check if all officer IDs exist in people table
    total_officers = tickets_df.select("officer_id").distinct().count()
    valid_officers = tickets_df.join(
        people_df.filter(col("profession") == "Police Officer"),
        tickets_df.officer_id == people_df.id,
        "left_semi"
    ).select("officer_id").distinct().count()
    
    print(f"Officer ID validation:")
    print(f"Total unique officers in tickets: {total_officers}")
    print(f"Valid officers found: {valid_officers}")
    print(f"Missing officers: {total_officers - valid_officers}")
    
    # Check if all license plates exist in automobiles table
    total_plates = tickets_df.select("license_plate").distinct().count()
    valid_plates = tickets_df.join(
        automobiles_df,
        tickets_df.license_plate == automobiles_df.license_plate,
        "left_semi"
    ).select("license_plate").distinct().count()
    
    print(f"\nLicense plate validation:")
    print(f"Total unique plates in tickets: {total_plates}")
    print(f"Valid plates found: {valid_plates}")
    print(f"Missing plates: {total_plates - valid_plates}")


def main():
    """Main function to orchestrate the data processing pipeline."""
    try:
        # Initialize Spark session
        spark = create_spark_session()

        # Get schema definitions
        ticket_schema = get_schema_definitions()

        # Load data
        tickets_df, people_df, automobiles_df = load_data(spark, ticket_schema)
        
        # Validate ticket data
        validated_tickets_df = validate_speeding_tickets(tickets_df)
        
        # Check relationships between datasets
        check_ticket_relationships(tickets_df, people_df, automobiles_df)

        
        # Continue with only valid records for further analysis
        valid_tickets_df = validated_tickets_df.filter(col("is_valid") == True)
        
        # Perform regular analyses with validated data
        analyze_officer_tickets(valid_tickets_df, people_df)
        analyze_ticket_trends(valid_tickets_df)
        analyze_ticket_costs(spark, valid_tickets_df, automobiles_df, people_df)

    except Exception as e:
        print(f"Error occurred: {str(e)}")
    finally:
        if 'spark' in locals():
            spark.stop()


if __name__ == "__main__":
    main()