import {Routes, Route} from "react-router-dom";

import Layout from "./components/Layout";
import NoMatch from "./pages/NoMatch";


import API from "./services/api.ts";
import Scrolling from "./pages/Scrolling";

export const api: API = new API();

function App() {
    return (
        <Routes>
            <Route path="/" element={<Layout/>}>
                <Route index element={<Scrolling/>}/>

                <Route path="*" element={<NoMatch/>}/>

            </Route>
        </Routes>
    );
}

export default App;
