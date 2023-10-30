from io import BytesIO
import json


def get_model_card_buf(model,
                       number_of_training_points,
                       number_of_validation_points,
                       graph_report_path,
                       input_type,
                       output_type):
    model_card = {
        "model_creation_date": model.date,
        "model_type": model.model_type,
        "model_path": model.file_path,
        "model_file_hash": model.model_hash,
        "input_type": input_type,
        "output_type": output_type,
        "number_of_training_points": number_of_training_points,
        "number_of_validation_points": number_of_validation_points,
        "training_loss": model.training_loss.item(),
        "validation_loss": model.validation_loss.item(),
        "graph_report": graph_report_path,
    }

    buf = BytesIO()
    buf.write(json.dumps(model_card, indent=4).encode())
    buf.seek(0)

    return buf
