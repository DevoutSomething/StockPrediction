import React from "react";
import { Route, Routes } from "react-router-dom";
import Header from "./components/Header.jsx";
import Home from "./pages/Home.jsx";
import Params from "./pages/Params.jsx";
import Stocks from "./pages/Stocks.jsx";

function App() {
  return (
    <div>
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/params" element={<Params />} />
        <Route path="/stocks" element={<Stocks />} />
      </Routes>
    </div>
  );
}

export default App;
