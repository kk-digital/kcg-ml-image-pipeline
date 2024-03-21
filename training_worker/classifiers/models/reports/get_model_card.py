from io import BytesIO
import json

def get_model_card_buf(classifier_name,
                       tag_id,
                       latest_model,
                       model_path,
                       creation_time):
    model_card = {
                "classifier_id": None,
                "classifier_name": classifier_name,
                "tag_id": tag_id,
                "model_sequence_number": None,
                "latest_model": latest_model,
                "model_path": model_path,
                "creation_time": creation_time
            }

    buf = BytesIO()
    buf.write(json.dumps(model_card, indent=4).encode())
    buf.seek(0)

    return buf, json.dumps(model_card, indent=4)