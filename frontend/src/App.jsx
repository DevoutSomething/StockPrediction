import { Route, Routes } from "react-router-dom"
import Home from "./pages/Home.jsx"
import Params from "./pages/Params.jsx"
import Stocks from "./pages/Stocks.jsx"

function App() {
  return (
    <div className="app">
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/params" element={<Params />} />
        <Route path="/stocks" element={<Stocks />} />
      </Routes>
    </div>
  )
}

export default App

