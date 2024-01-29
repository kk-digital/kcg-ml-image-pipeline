
def train_classifier(dataset_name: str,
                    minio_ip_addr=None,
                    minio_access_key=None,
                    minio_secret_key=None,
                    input_type="embedding",
                    tag_name=None,
                    ):
    date_now = datetime.now(tz=timezone("Asia/Hong_Kong")).strftime('%Y-%m-%d')
    print("Current datetime: {}".format(datetime.now(tz=timezone("Asia/Hong_Kong"))))
    bucket_name = "datasets"
    training_dataset_path = os.path.join(bucket_name, dataset_name)
    network_type = "elm-regression"
    output_type = "score"
    output_path = "{}/models/ranking".format(dataset_name)

    # check input type
    if input_type not in constants.ALLOWED_INPUT_TYPES:
        raise Exception("input type is not supported: {}".format(input_type))

    input_shape = 2 * 768
    if input_type in [constants.EMBEDDING_POSITIVE, constants.EMBEDDING_NEGATIVE, constants.CLIP]:
        input_shape = 768

    # load dataset
    # load images of tagged data
    # load untagged data random? same number?

    # get final filename
    sequence = 0
    filename = "{}-{:02}-{}-{}-{}".format(date_now, sequence, output_type, network_type, input_type)

    # if exist, increment sequence
    while True:
        filename = "{}-{:02}-{}-{}-{}".format(date_now, sequence, output_type, network_type, input_type)
        exists = cmd.is_object_exists(dataset_loader.minio_client, bucket_name,
                                      os.path.join(output_path, filename + ".safetensors"))
        if not exists:
            break

        sequence += 1

    # train model

    # save model
    # save model report
    # save graph report