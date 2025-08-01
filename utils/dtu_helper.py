def format_title(base_title: str, debug_status: bool) -> str:
    if debug_status:
        return f"{base_title} --- DEBUG MODE"
    return base_title