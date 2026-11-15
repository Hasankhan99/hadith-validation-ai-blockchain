package main

import (
	"encoding/json"
	"fmt"
	"time"

	"github.com/hyperledger/fabric-contract-api-go/contractapi"
)

// SmartContract defines the Hadith Validation Smart Contract
type SmartContract struct {
	contractapi.Contract
}

// ModelRegistration stores registered AI model metadata and weight SHA-256 hash
type ModelRegistration struct {
	ModelID        string  `json:"model_id"`
	ModelName      string  `json:"model_name"`
	SHA256Hash     string  `json:"sha256_hash"`
	AccuracyWeight float64 `json:"accuracy_weight"`
	Timestamp      string  `json:"timestamp"`
}

// ValidationRecord stores immutable transaction ledger entries for Hadith validation events
type ValidationRecord struct {
	TxID            string             `json:"tx_id"`
	HadithID        string             `json:"hadith_id"`
	HadithHash      string             `json:"hadith_hash"`
	ConsensusResult int                `json:"consensus_result"` // 1 = Sahih, 0 = Mawdu
	ConfidenceScore float64            `json:"confidence_score"`
	ModelVerdicts   map[string]int     `json:"model_verdicts"`
	ModelHashes     map[string]string  `json:"model_hashes"`
	Timestamp       string             `json:"timestamp"`
}

// RegisterModel registers an AI model and its SHA-256 weight hash
func (s *SmartContract) RegisterModel(ctx contractapi.TransactionContextInterface, modelID string, modelName string, hash string, weight float64) error {
	reg := ModelRegistration{
		ModelID:        modelID,
		ModelName:      modelName,
		SHA256Hash:     hash,
		AccuracyWeight: weight,
		Timestamp:      time.Now().Format(time.RFC3339),
	}

	regBytes, err := json.Marshal(reg)
	if err != nil {
		return err
	}

	return ctx.GetStub().PutState("MODEL_"+modelID, regBytes)
}

// VerifyAndRecordConsensus verifies model hashes and writes an immutable audit record to the ledger
func (s *SmartContract) VerifyAndRecordConsensus(
	ctx contractapi.TransactionContextInterface,
	txID string,
	hadithID string,
	hadithHash string,
	consensusResult int,
	confidence float64,
	verdictsJSON string,
	hashesJSON string,
) error {

	var verdicts map[string]int
	var hashes map[string]string

	if err := json.Unmarshal([]byte(verdictsJSON), &verdicts); err != nil {
		return fmt.Errorf("failed to parse verdicts JSON: %v", err)
	}

	if err := json.Unmarshal([]byte(hashesJSON), &hashes); err != nil {
		return fmt.Errorf("failed to parse hashes JSON: %v", err)
	}

	// Verify model hashes match registered values
	for modelID, submittedHash := range hashes {
		regBytes, err := ctx.GetStub().GetState("MODEL_" + modelID)
		if err != nil || regBytes == nil {
			return fmt.Errorf("model %s not registered on ledger", modelID)
		}
		var reg ModelRegistration
		json.Unmarshal(regBytes, &reg)
		if reg.SHA256Hash != submittedHash {
			return fmt.Errorf("SECURITY ALERT: Model %s weight hash mismatch! Tampering detected", modelID)
		}
	}

	record := ValidationRecord{
		TxID:            txID,
		HadithID:        hadithID,
		HadithHash:      hadithHash,
		ConsensusResult: consensusResult,
		ConfidenceScore: confidence,
		ModelVerdicts:   verdicts,
		ModelHashes:     hashes,
		Timestamp:       time.Now().Format(time.RFC3339),
	}

	recordBytes, err := json.Marshal(record)
	if err != nil {
		return err
	}

	return ctx.GetStub().PutState("VALIDATION_"+txID, recordBytes)
}

func main() {
	chaincode, err := contractapi.NewChaincode(&SmartContract{})
	if err != nil {
		fmt.Printf("Error creating Hadith validation chaincode: %s", err)
		return
	}

	if err := chaincode.Start(); err != nil {
		fmt.Printf("Error starting Hadith validation chaincode: %s", err)
	}
}
