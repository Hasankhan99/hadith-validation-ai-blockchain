package main

// Hadith validation chaincode (v2). Changes from v1 (blockchain/chaincode/hadith_validation.go):
//   - Timestamps come from the transaction proposal (GetTxTimestamp), not the peer clock, so all
//     endorsing peers produce identical write sets.
//   - A validation is accepted only if the submitted model set equals the full registered ensemble
//     and every digest matches (v1 checked only the models the client chose to submit).
//   - Unmarshal errors abort the transaction; duplicate validation IDs are rejected.
//   - Model registration is restricted to the administering organisation's MSP.
//   - Verdicts support the three-class scheme (0 = Mawdu, 1 = Da'if, 2 = Sahih) and record the
//     consensus status (VERIFIED / FLAGGED_FOR_HUMAN_REVIEW).

import (
	"encoding/json"
	"fmt"
	"sort"
	"time"

	"github.com/hyperledger/fabric-contract-api-go/v2/contractapi"
)

const (
	modelPrefix      = "MODEL_"
	validationPrefix = "VALIDATION_"
	adminMSP         = "Org1MSP"
)

var labelNames = map[int]string{0: "Mawdu", 1: "Daif", 2: "Sahih"}

type SmartContract struct {
	contractapi.Contract
}

type ModelRegistration struct {
	ModelID        string  `json:"model_id"`
	ModelName      string  `json:"model_name"`
	SHA256Hash     string  `json:"sha256_hash"`
	AccuracyWeight float64 `json:"accuracy_weight"`
	RegisteredBy   string  `json:"registered_by"`
	Timestamp      string  `json:"timestamp"`
}

type ValidationRecord struct {
	TxID            string            `json:"tx_id"`
	FabricTxID      string            `json:"fabric_tx_id"`
	HadithID        string            `json:"hadith_id"`
	HadithHash      string            `json:"hadith_hash"`
	ConsensusResult int               `json:"consensus_result"`
	ConsensusLabel  string            `json:"consensus_label"`
	ConfidenceScore float64           `json:"confidence_score"`
	Status          string            `json:"status"`
	ModelVerdicts   map[string]int    `json:"model_verdicts"`
	ModelHashes     map[string]string `json:"model_hashes"`
	Timestamp       string            `json:"timestamp"`
}

func txTime(ctx contractapi.TransactionContextInterface) (string, error) {
	ts, err := ctx.GetStub().GetTxTimestamp()
	if err != nil {
		return "", err
	}
	return time.Unix(ts.Seconds, int64(ts.Nanos)).UTC().Format(time.RFC3339Nano), nil
}

// RegisterModel registers (or re-registers) a model digest. Only the admin MSP may call it;
// every change remains visible through the key history.
func (s *SmartContract) RegisterModel(ctx contractapi.TransactionContextInterface, modelID, modelName, hash string, weight float64) error {
	msp, err := ctx.GetClientIdentity().GetMSPID()
	if err != nil {
		return err
	}
	if msp != adminMSP {
		return fmt.Errorf("access denied: %s may not register models", msp)
	}
	if len(hash) != 64 {
		return fmt.Errorf("invalid SHA-256 digest length %d", len(hash))
	}
	ts, err := txTime(ctx)
	if err != nil {
		return err
	}
	b, err := json.Marshal(ModelRegistration{modelID, modelName, hash, weight, msp, ts})
	if err != nil {
		return err
	}
	return ctx.GetStub().PutState(modelPrefix+modelID, b)
}

func registeredModels(ctx contractapi.TransactionContextInterface) (map[string]ModelRegistration, error) {
	it, err := ctx.GetStub().GetStateByRange(modelPrefix, modelPrefix+"~")
	if err != nil {
		return nil, err
	}
	defer it.Close()
	out := map[string]ModelRegistration{}
	for it.HasNext() {
		kv, err := it.Next()
		if err != nil {
			return nil, err
		}
		var reg ModelRegistration
		if err := json.Unmarshal(kv.Value, &reg); err != nil {
			return nil, fmt.Errorf("corrupt registration %s: %v", kv.Key, err)
		}
		out[reg.ModelID] = reg
	}
	return out, nil
}

// VerifyAndRecordConsensus checks the submitted digests against the full registered ensemble and,
// only if all match, writes an immutable validation record.
func (s *SmartContract) VerifyAndRecordConsensus(ctx contractapi.TransactionContextInterface,
	txID, hadithID, hadithHash string, consensusResult int, confidence float64, status, verdictsJSON, hashesJSON string) error {

	var verdicts map[string]int
	var hashes map[string]string
	if err := json.Unmarshal([]byte(verdictsJSON), &verdicts); err != nil {
		return fmt.Errorf("failed to parse verdicts JSON: %v", err)
	}
	if err := json.Unmarshal([]byte(hashesJSON), &hashes); err != nil {
		return fmt.Errorf("failed to parse hashes JSON: %v", err)
	}
	if _, ok := labelNames[consensusResult]; !ok {
		return fmt.Errorf("invalid consensus result %d", consensusResult)
	}
	if confidence < 0 || confidence > 1 {
		return fmt.Errorf("confidence %f outside [0,1]", confidence)
	}

	regs, err := registeredModels(ctx)
	if err != nil {
		return err
	}
	if len(regs) == 0 {
		return fmt.Errorf("no models registered")
	}
	for id := range regs {
		if _, ok := hashes[id]; !ok {
			return fmt.Errorf("SECURITY ALERT: registered model %s missing from submission", id)
		}
	}
	ids := make([]string, 0, len(hashes))
	for id := range hashes {
		ids = append(ids, id)
	}
	sort.Strings(ids) // deterministic error order across endorsers
	for _, id := range ids {
		reg, ok := regs[id]
		if !ok {
			return fmt.Errorf("SECURITY ALERT: model %s not registered on ledger", id)
		}
		if reg.SHA256Hash != hashes[id] {
			return fmt.Errorf("SECURITY ALERT: model %s weight hash mismatch, tampering detected", id)
		}
		if _, ok := verdicts[id]; !ok {
			return fmt.Errorf("missing verdict for model %s", id)
		}
	}

	key := validationPrefix + txID
	existing, err := ctx.GetStub().GetState(key)
	if err != nil {
		return err
	}
	if existing != nil {
		return fmt.Errorf("validation %s already recorded", txID)
	}
	ts, err := txTime(ctx)
	if err != nil {
		return err
	}
	rec := ValidationRecord{txID, ctx.GetStub().GetTxID(), hadithID, hadithHash, consensusResult, labelNames[consensusResult],
		confidence, status, verdicts, hashes, ts}
	b, err := json.Marshal(rec)
	if err != nil {
		return err
	}
	return ctx.GetStub().PutState(key, b)
}

func (s *SmartContract) GetValidation(ctx contractapi.TransactionContextInterface, txID string) (*ValidationRecord, error) {
	b, err := ctx.GetStub().GetState(validationPrefix + txID)
	if err != nil || b == nil {
		return nil, fmt.Errorf("validation %s not found", txID)
	}
	var rec ValidationRecord
	if err := json.Unmarshal(b, &rec); err != nil {
		return nil, err
	}
	return &rec, nil
}

func (s *SmartContract) GetModel(ctx contractapi.TransactionContextInterface, modelID string) (*ModelRegistration, error) {
	b, err := ctx.GetStub().GetState(modelPrefix + modelID)
	if err != nil || b == nil {
		return nil, fmt.Errorf("model %s not found", modelID)
	}
	var reg ModelRegistration
	if err := json.Unmarshal(b, &reg); err != nil {
		return nil, err
	}
	return &reg, nil
}

func main() {
	cc, err := contractapi.NewChaincode(&SmartContract{})
	if err != nil {
		fmt.Printf("Error creating Hadith validation chaincode: %s", err)
		return
	}
	if err := cc.Start(); err != nil {
		fmt.Printf("Error starting Hadith validation chaincode: %s", err)
	}
}
