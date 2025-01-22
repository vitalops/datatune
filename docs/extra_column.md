# extra_column

## extra_column.ColumnType

Enumeration of supported data types for extra columns.

Attributes:
    INT: Integer type
    FLOAT: Floating-point number type
    STR: String type
    BOOL: Boolean type
    LABEL: Categorical label type

## extra_column.ExtraColumn

A class representing an additional column added to a view in Datatune.

Args:
    id (str): The unique identifier of the column
    view (View): The view object that contains this column

Attributes:
    id (str): The column's unique identifier
    view (View): The view containing this column

## extra_column.entity

Returns the entity associated with this column's view.

## extra_column.workspace

Returns the workspace containing this column's view.

## extra_column.name

Returns the name of the column.

Returns:
    str: The column's name

## extra_column.type

Returns the data type of the column.

Returns:
    ColumnType: The column's data type (INT, FLOAT, STR, BOOL, or LABEL)

## extra_column.api

Returns the API instance associated with this column's view.

## extra_column.labels

Returns the list of possible labels for categorical columns.

Returns:
    List[str]: List of valid labels for this column if it's of type LABEL,
              otherwise returns an empty list

