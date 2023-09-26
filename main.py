import multiprocessing
import uvicorn


if __name__ == "__main__":
    # get number of cores
    cores = multiprocessing.cpu_count()

    # Run the API
    uvicorn.run("orchestration.api.main:app", host="127.0.0.1", port=8000, workers=cores, reload=True)
