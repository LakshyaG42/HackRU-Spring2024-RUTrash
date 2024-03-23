import './App.css';
// import BillInput from './pageNew';

import React, { useState } from "react";
 
function App() {
    const [file, setFile] = useState();
    function handleChange(e) {
        console.log(e.target.files);
        setFile(URL.createObjectURL(e.target.files[0]));
    }
 
    return (
        <div className="App">
            <h2>Add Image:</h2>
            <input type="file" onChange={handleChange} />
            <img src={file} />
        </div>
    );
}

// function App() {
//   const handleChange = (event) => {
//     const milesTraveledInCar = event.target.value;
//     <p>
//       Carbon Emissions: {milesTraveledInCar * 0.79}
//     </p>
//   };

//   return (
//     <div className="App">
//       {/* <header className="App-header">
//         <p>
//           Enter your yearly mileage:
//         </p>
//         <input type="range" min="0" max="30000" step="5000" onChange={handleChange} style={{ width: '50%' }} />
//       </header> */}
//       <BillInput />
//     </div>
//   );
// }


export default App;