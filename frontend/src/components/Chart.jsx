import { useState } from "react";
import "./Styles/chart.css";
import { FaChevronUp, FaChevronDown } from "react-icons/fa";

export default function Chart({ dataArr }) {
  const [data, setData] = useState(dataArr);
  const [sortedByName, setSortedByName] = useState(false);
  const [sortedByPrice, setSortedByPrice] = useState(false);
  const [sortedByRisk, setSortedByRisk] = useState(false);
  const [sortedByProfit, setSortedByProfit] = useState(false);

  const sortData = (key, setSortedBy, sortedBy) => {
    const sortedData = [...data];

    sortedData.sort((a, b) => {
      if (a[key] < b[key]) return sortedBy ? 1 : -1;
      if (a[key] > b[key]) return sortedBy ? -1 : 1;
      return 0;
    });

    setData(sortedData);
    setSortedBy(!sortedBy);
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
                <button onClick={() => sortData("name", setSortedByName, sortedByName)}>
                  Name {sortedByName ? <FaChevronUp /> : <FaChevronDown />}
                </button>
              </th>
              <th>
                <button onClick={() => sortData("price", setSortedByPrice, sortedByPrice)}>
                  Price {sortedByPrice ? <FaChevronUp /> : <FaChevronDown />}
                </button>
              </th>
              <th>
                <button onClick={() => sortData("profit", setSortedByProfit, sortedByProfit)}>
                  Profit {sortedByProfit ? <FaChevronUp /> : <FaChevronDown />}
                </button>
              </th>
              <th>
                <button onClick={() => sortData("risk", setSortedByRisk, sortedByRisk)}>
                  Risk {sortedByRisk ? <FaChevronUp /> : <FaChevronDown />}
                </button>
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
