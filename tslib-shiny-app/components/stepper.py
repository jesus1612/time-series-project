# Stepper component for wizard navigation
from shiny import ui, reactive
from typing import List, Dict, Any

class StepperComponent:
    """Wizard/Stepper component for step-by-step navigation"""
    
    def __init__(self, steps: List[Dict[str, str]]):
        """
        Initialize stepper with list of steps
        
        Args:
            steps: List of dictionaries with 'title' and 'description' keys
        """
        self.steps = steps
        self.current_step = 0
        self.total_steps = len(steps)
    
    def render_header(self) -> ui.Tag:
        """Render stepper header with title and progress"""
        return ui.div(
            ui.div(
                ui.tags.h2(
                    self.steps[self.current_step]["title"],
                    class_="stepper-title"
                ),
                ui.div(
                    f"Paso {self.current_step + 1} de {self.total_steps}",
                    class_="stepper-progress"
                ),
                class_="stepper-header"
            ),
            class_="stepper-container"
        )
    
    def render_navigation(self) -> ui.Tag:
        """Render navigation buttons"""
        return ui.div(
            ui.div(
                ui.input_action_button(
                    "prev_step",
                    "← Anterior",
                    class_="btn btn-secondary"
                ) if self.current_step > 0 else ui.div(),
                class_="d-flex"
            ),
            ui.div(
                ui.input_action_button(
                    "next_step",
                    "Siguiente →" if self.current_step < self.total_steps - 1 else "Finalizar",
                    class_="btn btn-primary"
                ),
                class_="d-flex"
            ),
            class_="stepper-navigation"
        )
    
    def render_step_indicator(self) -> ui.Tag:
        """Render visual step indicator"""
        indicators = []
        for i, step in enumerate(self.steps):
            is_active = i == self.current_step
            is_completed = i < self.current_step
            
            indicator_class = "step-indicator"
            if is_active:
                indicator_class += " active"
            elif is_completed:
                indicator_class += " completed"
            
            indicators.append(
                ui.div(
                    ui.div(
                        str(i + 1),
                        class_="step-number"
                    ),
                    ui.div(
                        ui.div(step["title"], class_="step-title"),
                        ui.div(step["description"], class_="step-description"),
                        class_="step-content"
                    ),
                    class_=indicator_class
                )
            )
        
        return ui.div(
            *indicators,
            class_="step-indicators"
        )
    
    def get_current_step_content(self, content_func) -> ui.Tag:
        """Get content for current step"""
        return ui.div(
            content_func(),
            class_="stepper-content"
        )
    
    def can_go_next(self) -> bool:
        """Check if can proceed to next step"""
        return self.current_step < self.total_steps - 1
    
    def can_go_previous(self) -> bool:
        """Check if can go to previous step"""
        return self.current_step > 0
    
    def go_next(self):
        """Move to next step"""
        if self.can_go_next():
            self.current_step += 1
    
    def go_previous(self):
        """Move to previous step"""
        if self.can_go_previous():
            self.current_step -= 1
    
    def is_last_step(self) -> bool:
        """Check if current step is the last one"""
        return self.current_step == self.total_steps - 1
    
    def get_step_info(self) -> Dict[str, Any]:
        """Get current step information"""
        return {
            "current": self.current_step,
            "total": self.total_steps,
            "title": self.steps[self.current_step]["title"],
            "description": self.steps[self.current_step]["description"],
            "is_first": self.current_step == 0,
            "is_last": self.is_last_step()
        }
