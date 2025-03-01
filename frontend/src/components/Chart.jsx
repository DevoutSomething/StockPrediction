"use client";

import { useState } from "react";
import "./Styles/chart.css";

export default function Chart({ dataArr }) {
  const [data, setData] = useState(dataArr);
  const [sortedByName, setSortedByName] = useState(false);
  const [sortedByPrice, setSortedByPrice] = useState(false);
  const [sortedByRisk, setSortedByRisk] = useState(false);
  const [sortedByProfit, setSortedByProfit] = useState(false);

  const sortDataName = () => {
    const sortedData = [...data];

    sortedData.sort((a, b) => {
      if (a.name < b.name) {
        return sortedByName ? 1 : -1;
      }
      if (a.name > b.name) {
        return sortedByName ? -1 : 1;
      }
      return 0;
    });

    setData(sortedData);
    setSortedByName(!sortedByName);
  };

  const sortDataPrice = () => {
    const sortedData = [...data];

    sortedData.sort((a, b) => {
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

  const sortDataRisk = () => {
    const sortedData = [...data];

    sortedData.sort((a, b) => {
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

  const sortDataProfit = () => {
    const sortedData = [...data];

    sortedData.sort((a, b) => {
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

  const getRiskClass = (risk) => {
    if (risk <= 0.3) return "veryRisky";
    if (risk <= 0.6) return "mediumRisk";
    return "lowRisk";
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
                <button onClick={sortDataProfit}>Profit</button>
              </th>
              <th>
                <button onClick={sortDataRisk}>Risk</button>
              </th>
            </tr>
          </thead>
          <tbody>
            {data.map((stock, index) => (
              <tr key={index}>
                <td>
                  <a
                    href={`https://www.nasdaq.com/market-activity/stocks/${stock.code}`}
                    target="_blank"
                    rel="noopener noreferrer"
                  >
                    {stock.name}
                  </a>
                </td>
                <td>{stock.price}</td>
                <td>{stock.profit}</td>
                <td className={getRiskClass(stock.risk)}>{stock.risk}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
