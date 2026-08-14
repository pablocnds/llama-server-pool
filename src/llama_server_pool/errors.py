class PoolError(Exception):
    status_code = 500
    code = "pool_error"

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class ModelNotFoundError(PoolError):
    status_code = 404
    code = "model_not_found"


class DuplicateModelError(PoolError):
    status_code = 409
    code = "duplicate_model"


class ModelConflictError(PoolError):
    status_code = 409
    code = "model_conflict"


class CapacityError(PoolError):
    status_code = 507
    code = "insufficient_capacity"


class StartupError(PoolError):
    status_code = 502
    code = "model_startup_failed"


class PoolShuttingDownError(PoolError):
    status_code = 503
    code = "pool_shutting_down"


class ProfileStoreError(PoolError):
    status_code = 500
    code = "profile_store_error"
