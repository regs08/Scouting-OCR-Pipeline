from utils.components.confusion_matrix_component import ConfusionMatrixComponent
from utils.components.confusion_matrix_visualizer_component import ConfusionMatrixVisualizerComponent
from utils.components.error_analysis_visualizer_component import ErrorAnalysisVisualizerComponent
from utils.managers.confusion_matrix_manager import ConfusionMatrixManager
from utils.runnable_component_config import RunnableComponentConfig

confusion_matrix_component_config = RunnableComponentConfig(
    component_class=ConfusionMatrixComponent,
    checkpoint_name="ckpt1_confusion_matrix",
    checkpoint_number=1,
    description="Generate confusion matrix analysis"
)

visualizer_config = RunnableComponentConfig(
    component_class=ConfusionMatrixVisualizerComponent,
    checkpoint_name="ckpt2_visualization",
    checkpoint_number=2,
    description="Create confusion matrix visualizations"
)

error_analysis_config = RunnableComponentConfig(
    component_class=ErrorAnalysisVisualizerComponent,
    checkpoint_name="ckpt3_error_analysis",
    checkpoint_number=3,
    description="Analyze and visualize error patterns"
)


confusion_matrix_config = RunnableComponentConfig(
    component_class=ConfusionMatrixManager,
    checkpoint_name="ckpt4_confusion_matrix_setup",
    checkpoint_number=4,
    description="Confusion Matrix Setup and Analysis",

    component_configs=[
        confusion_matrix_component_config,
        visualizer_config,
        error_analysis_config   
    ]
)