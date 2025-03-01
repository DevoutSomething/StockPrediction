import "./styles/stocksTable.css";
import { useState } from "react";

export default function Chart({ dataArr }) {
  const [data, setData] = useState(dataArr);
  const [sortedByName, setSortedByName] = useState(false);
  const [sortedByPrice, setSortedByPrice] = useState(false);
  const [sortedByRisk, setSortedByRisk] = useState(false);
  const [sortedByProfit, setSortedByProfit] = useState(false);

  const sortDataName = function () {
    // Create a copy of the data to avoid direct mutation
    const sortedData = [...data];

    sortedData.sort((a, b) => {
      // Compare based on name instead of age, assuming name is the property to sort by
      if (a.name < b.name) {
        return sortedByName ? 1 : -1;
      }
      if (a.name > b.name) {
        return sortedByName ? -1 : 1;
      }
      return 0;
    });

    // Update the state with the sorted data and toggle the sorting direction
    setData(sortedData);
    setSortedByName(!sortedByName);
  };
  const sortDataPrice = function () {
    const sortedData = [...data];

    sortedData.sort((a, b) => {
      // Compare based on name instead of age, assuming name is the property to sort by
      if (a.price < b.price) {
        return sortedByPrice ? 1 : -1;
      }
      if (a.price > b.price) {
        return sortedByPrice ? -1 : 1;
      }
      return 0;
    });

    setData(sortedData);
    setSortedByPrice(!sortedByPrice);
  };

  const sortDataRisk = function () {
    const sortedData = [...data];

    sortedData.sort((a, b) => {
      // Compare based on name instead of age, assuming name is the property to sort by
      if (a.risk < b.risk) {
        return sortedByRisk ? 1 : -1;
      }
      if (a.risk > b.risk) {
        return sortedByRisk ? -1 : 1;
      }
      return 0;
    });

    setData(sortedData);
    setSortedByRisk(!sortedByRisk);
  };

  const sortDataProfit = function () {
    const sortedData = [...data];

    sortedData.sort((a, b) => {
      // Compare based on name instead of age, assuming name is the property to sort by
      if (a.profit < b.profit) {
        return sortedByProfit ? 1 : -1;
      }
      if (a.profit > b.profit) {
        return sortedByProfit ? -1 : 1;
      }
      return 0;
    });

    setData(sortedData);
    setSortedByProfit(!sortedByProfit);
  };

  return (
    <div>
      <div className="table-container">
        <table>
          <thead>
            <tr>
              <th>
                <button onClick={sortDataName}>Name</button>
              </th>
              <th>
                <button onClick={sortDataPrice}>Price</button>
              </th>
              <th>
                <button onClick={sortDataRisk}>Risk</button>
              </th>
              <th>
                <button onClick={sortDataProfit}>Profit</button>
              </th>
            </tr>
          </thead>
          <tbody>
            {data.map((stock, index) => (
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
  );
}
