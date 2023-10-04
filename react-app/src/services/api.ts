// api.ts

import axios, {AxiosInstance, AxiosResponse} from 'axios';


class API {
    public baseUrl: string;
    private instance: AxiosInstance;

    constructor() {
        this.baseUrl = import.meta.env.VITE_API_URL || 'http://192.168.3.1:8111';

        this.instance = axios.create({
            baseURL: this.baseUrl,
        });
    }


    async get<T = any>(endpoint: string, params?: object): Promise<AxiosResponse<T>> {
        return this.instance.get<T>(endpoint, {params});
    }

    async post<T = any>(endpoint: string, body: object): Promise<AxiosResponse<T>> {
        return this.instance.post<T>(endpoint, body);
    }

    async put<T = any>(endpoint: string, body: object): Promise<AxiosResponse<T>> {
        return this.instance.put<T>(endpoint, body);
    }

    async delete<T = any>(endpoint: string): Promise<AxiosResponse<T>> {
        return this.instance.delete<T>(endpoint);
    }
}

export default API;
