import "./styles/stocksTable.css"

export default function Chart({ dataArr }) {
  return (
    <div>
      <div className="table-container">
        <table>
          <thead>
            <tr>
              <th>Name</th>
              <th>Price</th>
              <th>Risk</th>
              <th>Profit</th>
            </tr>
          </thead>
          <tbody>
            {dataArr.map((stock, index) => (
              <tr key={index}>
                <td>{stock.name}</td>
                <td>{stock.price}</td>
                <td>{stock.risk}</td>
                <td>{stock.profit}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}

