from typing import Dict


class DatatuneException(Exception):
    """Generic DatatuneException class for all datatune errors"""

    code = None

    def __init__(self, message, errcode=None):
        if not message:
            message = type(self).__name__
        self.message = message

        if errcode:
            self.code = errcode

        super().__init__(
            f"<Response [{self.code}]> {message}"
            if self.code
            else f"<Response> {message}"
        )


class DatatuneBadRequest(DatatuneException):
    """400 - Bad Request -- The request was unacceptable, often due to missing a required parameter."""

    code = 400


class DatatuneUnauthorized(DatatuneException):
    """401 - Unauthorized -- No valid API key provided or the API key did not have enough privileges."""

    code = 401


class DatatunePaymentRequired(DatatuneException):
    """402 - Payment Required -- Payment is required for this request."""

    code = 402


class DatatuneNotFound(DatatuneException):
    """404 - Not Found -- The requested resource doesn't exist."""

    code = 404


class DatatuneConflict(DatatuneException):
    """409 - Conflict -- The request could not be completed due to a conflict, such as an existing resource."""

    code = 409


class DatatuneTooManyRequests(DatatuneException):
    """429 - Too Many Requests -- Too many requests hit the API too quickly. Please retry your request after some time."""

    code = 429


class DatatuneInternalServerError(DatatuneException):
    """500 - Internal Server Error -- An error occurred on our servers while processing the request."""

    code = 500


class DatatuneServiceUnavailable(DatatuneException):
    """503 - Service Unavailable -- The server is currently unavailable (because it is overloaded or down for maintenance)."""

    code = 503


class DatatuneGatewayTimeout(DatatuneException):
    """504 - Gateway Timeout -- The server was acting as a gateway or proxy and did not receive a timely response from the upstream server."""

    code = 504


ExceptionMap: Dict[int, DatatuneException] = {
    DatatuneBadRequest.code: DatatuneBadRequest,
    DatatuneUnauthorized.code: DatatuneUnauthorized,
    DatatunePaymentRequired.code: DatatunePaymentRequired,
    DatatuneNotFound.code: DatatuneNotFound,
    DatatuneConflict.code: DatatuneConflict,
    DatatuneTooManyRequests.code: DatatuneTooManyRequests,
    DatatuneInternalServerError.code: DatatuneInternalServerError,
    DatatuneServiceUnavailable.code: DatatuneServiceUnavailable,
    DatatuneGatewayTimeout.code: DatatuneGatewayTimeout,
}
