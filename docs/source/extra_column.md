
# extra_column


## extra_column.ColumnType


## extra_column.ExtraColumn


**Args:**

* **id (str)** - The unique identifier of the column
* **view (View)** - The view object that contains this column
* **Attributes** - 
* **id (str)** - The column's unique identifier
* **view (View)** - The view containing this column

## extra_column.entity


## extra_column.workspace


## extra_column.name


**Returns:**

str: The column's name

## extra_column.type


**Returns:**

ColumnType: The column's data type (INT, FLOAT, STR, BOOL, or LABEL)

## extra_column.api


## extra_column.labels


**Returns:**

List[str]: List of valid labels for this column if it's of type LABEL,
otherwise returns an empty list