import pathway as pw

class InputSchema(pw.Schema):
    name: str
    age: int

# Sample input table
data = [("Alice", 23), ("Bob", 30), ("Charlie", 27)]
input_table = pw.debug.table_from_rows(InputSchema, data)

# Correct way to select and transform columns
output_table = input_table.select(
    name=pw.this.name,
    next_age=pw.this.age + 1
)

# Compute and print results
pw.debug.compute_and_print(output_table)