import React, { useState } from 'react';
import './pageNew.css';

    const BillInput = () => {
      const [electricBill, setElectricBill] = useState('');
      const [gasBill, setGasBill] = useState('');
    
      const handleElectricBillChange = (e) => {
        setElectricBill(e.target.value);
      };
    
      const handleGasBillChange = (e) => {
        setGasBill(e.target.value);
      };
    
      return (
        <>
        <div className="energy-usage-container">
          <div className="box">
            <h2>Energy Usage</h2>
            <h5>What is your monthly bill?</h5>
            <div className="input-wrapper">
              <label htmlFor="electricBill">Electric Bill:</label>
              <input
                type="number"
                id="electricBill"
                value={electricBill}
                onChange={handleElectricBillChange}
                placeholder="Enter electric bill"
              />
            </div>
            <div className="input-wrapper">
              <label htmlFor="gasBill">Gas Bill:</label>
              <input
                type="number"
                id="gasBill"
                value={gasBill}
                onChange={handleGasBillChange}
                placeholder="Enter gas bill"
              />
            </div>
            <div>
                <input type="range" min="0" max="30000" step="5000" style={{ width: '50%' }} />
            </div>
          </div>
        </div>
          <style jsx>{`
            
            .energy-usage-container {
            border-radius: 30px;
            border-color: rgba(0, 0, 0, 1);
            border-style: solid;
            border-width: 1px;
            background-color: #fff;
            display: flex;
            flex-direction: column;
            align-items: center;
            width: 100%;
            padding: 10px 10px 50px 10px;
            }
            @media (max-width: 991px) {
            .energy-usage-container {
                max-width: 100%;
                margin-top: 24px;
                padding: 0 20px;
            }
            }
            // .div-4 {
            // display: flex;
            // width: 343px;
            // max-width: 100%;
            // flex-direction: column;
            // align-items: center;
            // }
            // .div-5 {
            // color: #000;
            // text-align: center;
            // font: 700 35px Kefa, sans-serif;
            // }
            // .div-6 {
            // color: #000;
            // text-align: center;
            // margin-top: 14px;
            // font: 400 22px Kefa, sans-serif;
            // }
            // .div-7 {
            // align-self: stretch;
            // display: flex;
            // margin-top: 27px;
            // gap: 11px;
            // }
            // .div-8 {
            // color: #000;
            // margin: auto 0;
            // font: 400 22px Kefa, sans-serif;
            // }
            // .div-9 {
            // display: flex;
            // flex-direction: column;
            // flex: 1;
            // }
            // .div-10 {
            // border-color: rgba(0, 0, 0, 1);
            // border-style: solid;
            // border-width: 1px;
            // background-color: #fff;
            // height: 29px;
            // }
            // .div-11 {
            // background-color: rgba(255, 255, 255, 1);
            // border-color: rgba(0, 0, 0, 1);
            // border-style: solid;
            // border-width: 1px;
            // margin-top: 42px;
            // height: 29px;
            // }
            // @media (max-width: 991px) {
            // .div-11 {
            //     margin-top: 40px;
            // }
            // }
            // .column-2 {
            // display: flex;
            // flex-direction: column;
            // line-height: normal;
            // width: 50%;
            // margin-left: 20px;
            // }
            // @media (max-width: 991px) {
            // .column-2 {
            //     width: 100%;
            // }
            // }
            // .div-12 {
            // border-radius: 30px;
            // border-color: rgba(0, 0, 0, 1);
            // border-style: solid;
            // border-width: 1px;
            // background-color: #fff;
            // display: flex;
            // flex-direction: column;
            // align-items: center;
            // width: 100%;
            // padding: 27px 56px 47px 10px;
            // }
            // @media (max-width: 991px) {
            // .div-12 {
            //     max-width: 100%;
            //     margin-top: 24px;
            //     padding-right: 20px;
            // }
            // }
            // .div-13 {
            // color: #000;
            // text-align: center;
            // font: 700 35px Kefa, sans-serif;
            // }
            // .div-14 {
            // color: #000;
            // text-align: center;
            // margin-top: 14px;
            // font: 400 22px Kefa, sans-serif;
            // }
            // .div-15 {
            // align-self: stretch;
            // margin-top: 11px;
            // }
            // @media (max-width: 991px) {
            // .div-15 {
            //     max-width: 100%;
            // }
            // }
            // .div-16 {
            // gap: 20px;
            // display: flex;
            // }
            // @media (max-width: 991px) {
            // .div-16 {
            //     flex-direction: column;
            //     align-items: stretch;
            //     gap: 0px;
            // }
            // }
            // .column-3 {
            // display: flex;
            // flex-direction: column;
            // line-height: normal;
            // width: 66%;
            // margin-left: 0px;
            // }
            // @media (max-width: 991px) {
            // .column-3 {
            //     width: 100%;
            // }
            // }
            // .div-17 {
            // color: #000;
            // font: 400 22px Kefa, sans-serif;
            // }
            // @media (max-width: 991px) {
            // .div-17 {
            //     margin-top: 7px;
            // }
            // }
            // .column-4 {
            // display: flex;
            // flex-direction: column;
            // line-height: normal;
            // width: 34%;
            // margin-left: 20px;
            // }
            // @media (max-width: 991px) {
            // .column-4 {
            //     width: 100%;
            // }
            // }
            // .div-18 {
            // display: flex;
            // flex-grow: 1;
            // flex-direction: column;
            // }
            // @media (max-width: 991px) {
            // .div-18 {
            //     margin-top: 7px;
            // }
            // }
            // .div-19 {
            // border-color: rgba(0, 0, 0, 1);
            // border-style: solid;
            // border-width: 1px;
            // background-color: #fff;
            // height: 29px;
            // }
            // .div-20 {
            // background-color: rgba(255, 255, 255, 1);
            // border-color: rgba(0, 0, 0, 1);
            // border-style: solid;
            // border-width: 1px;
            // margin-top: 19px;
            // height: 29px;
            // }
            // .div-21 {
            // background-color: rgba(255, 255, 255, 1);
            // border-color: rgba(0, 0, 0, 1);
            // border-style: solid;
            // border-width: 1px;
            // margin-top: 23px;
            // height: 29px;
            // }
        `}</style> 
        </>
      );
    };
    
export default BillInput;