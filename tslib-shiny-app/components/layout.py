# Layout components for consistent app structure
from shiny import ui

def create_app_layout(title: str, subtitle: str = None) -> ui.Tag:
    """Create main app layout with header and container"""
    
    header = ui.div(
        ui.div(
            ui.tags.h1(title, class_="app-title"),
            ui.tags.p(subtitle, class_="app-subtitle") if subtitle else None,
            class_="container-fluid"
        ),
        class_="app-header"
    )
    
    return ui.div(
        header,
        ui.div(
            class_="main-container"
        ),
        class_="app-wrapper"
    )

def create_card(title: str, content: ui.Tag, subtitle: str = None) -> ui.Tag:
    """Create a card component with title and content"""
    
    return ui.div(
        ui.div(
            ui.tags.h3(title, class_="card-title"),
            ui.tags.p(subtitle, class_="card-subtitle") if subtitle else None,
            class_="card-header"
        ),
        ui.div(content, class_="card-body"),
        class_="card"
    )

def create_metric_card(value: str, label: str, icon: str = None) -> ui.Tag:
    """Create a metric display card"""
    
    return ui.div(
        ui.div(
            ui.div(icon, class_="metric-icon") if icon else None,
            ui.div(value, class_="metric-value"),
            ui.tags.p(label, class_="metric-label"),
            class_="metric-content"
        ),
        class_="metric-card"
    )

def create_form_group(label: str, control: ui.Tag, help_text: str = None) -> ui.Tag:
    """Create a form group with label and control"""
    
    return ui.div(
        ui.tags.label(label, class_="form-label"),
        control,
        ui.tags.small(help_text, class_="form-help") if help_text else None,
        class_="form-group"
    )

def create_status_badge(text: str, status: str = "info") -> ui.Tag:
    """Create a status badge"""
    
    return ui.span(
        text,
        class_=f"status-indicator status-{status}"
    )

def create_progress_bar(value: int, max_value: int = 100, label: str = None) -> ui.Tag:
    """Create a progress bar"""
    
    percentage = (value / max_value) * 100 if max_value > 0 else 0
    
    return ui.div(
        ui.div(
            ui.div(
                class_="progress-bar",
                style=f"width: {percentage}%"
            ),
            class_="progress-container"
        ),
        ui.div(label, class_="progress-label") if label else None,
        class_="progress-wrapper"
    )

def create_data_table(data: list, headers: list = None) -> ui.Tag:
    """Create a data table"""
    
    if headers is None:
        headers = [f"Columna {i+1}" for i in range(len(data[0]) if data else 0)]
    
    table_header = ui.tags.tr(
        *[ui.tags.th(header) for header in headers]
    )
    
    table_rows = []
    for row in data:
        table_rows.append(
            ui.tags.tr(
                *[ui.tags.td(str(cell)) for cell in row]
            )
        )
    
    return ui.div(
        ui.tags.table(
            ui.tags.thead(table_header),
            ui.tags.tbody(*table_rows),
            class_="data-table"
        ),
        class_="table-container"
    )

def create_file_upload_area(
    input_id: str,
    label: str = "Seleccionar archivo",
    accept: str = ".csv,.xlsx,.xls",
    multiple: bool = False
) -> ui.Tag:
    """Create a file upload area with hidden input and custom drag & drop area"""
    
    return ui.div(
        ui.div(
            ui.input_file(
                input_id,
                label,
                accept=accept,
                multiple=multiple
            ),
            class_="file-input-hidden"
        ),
        ui.div(
            ui.div("📁", class_="file-upload-icon"),
            ui.div("Arrastra y suelta tu archivo aquí", class_="file-upload-text"),
            ui.div("o haz clic para seleccionar", class_="file-upload-hint"),
            class_="file-upload-area",
            onclick=f"document.getElementById('{input_id}').click()"
        ),
        class_="file-upload-wrapper"
    )

def create_action_buttons(
    buttons: list,
    class_name: str = "btn-group"
) -> ui.Tag:
    """Create a group of action buttons"""
    
    button_elements = []
    for button in buttons:
        button_elements.append(
            ui.input_action_button(
                button["id"],
                button["label"],
                class_=f"btn {button.get('class', 'btn-secondary')}"
            )
        )
    
    return ui.div(
        *button_elements,
        class_=class_name
    )
