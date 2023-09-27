#!/bin/bash

curl -X 'POST' http://192.168.3.1:8111/add-job -H "Content-Type: application/json" -d "{
        'task_type': 'inpainting_generation_task'
    }"
