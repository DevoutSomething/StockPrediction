import "./Styles/body.css";

export default function Body() {
  return (
    <div className="body-container">
      <div className="section-title">
        <h1>Description</h1>
        <div className="underline"></div>
      </div>

      <div className="text-content">
        <p>
          Lorem ipsum dolor sit amet consectetur adipisicing elit. Culpa
          adipisci perspiciatis laborum officia et consequuntur, sit, blanditiis
          soluta, provident id dolore architecto aliquid voluptatibus est ipsum
          sunt dolorum! Dolore, minima!
        </p>
        <p>
          Lorem ipsum dolor sit amet consectetur adipisicing elit. Culpa
          adipisci perspiciatis laborum officia et consequuntur, sit, blanditiis
          soluta, provident id dolore architecto aliquid voluptatibus est ipsum
          sunt dolorum! Dolore, minima!
        </p>

        <div className="feature-cards">
          <div className="feature-card">
            <div className="card-icon">📈</div>
            <h3>Accurate Predictions</h3>
            <p>
              Our algorithm provides highly accurate stock predictions based on
              historical data and market trends.
            </p>
          </div>

          <div className="feature-card">
            <div className="card-icon">⚡</div>
            <h3>Real-time Updates</h3>
            <p>
              Get real-time updates and notifications about significant market
              changes affecting your stocks.
            </p>
          </div>
        </div>

        <p>
          Lorem ipsum dolor sit amet consectetur adipisicing elit. Culpa
          adipisci perspiciatis laborum officia et consequuntur, sit, blanditiis
          soluta, provident id dolore architecto aliquid voluptatibus est ipsum
          sunt dolorum! Dolore, minima!
        </p>
      </div>
    </div>
  );
}
