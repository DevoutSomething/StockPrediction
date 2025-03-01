import "./styles/stockBody.css"
import Chart from "./Chart"
import ChartHeader from "./ChartHeader"

export default function StockBody({ dataArr }) {
  return (
    <div className="stock-body">
      <ChartHeader />
      <Chart dataArr={dataArr} />
    </div>
  )
}

