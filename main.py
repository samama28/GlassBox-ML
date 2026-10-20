
from typing import List

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Import the Phase 1 neural network engine
from GlassBox_Engine import GlassBoxMLP


# ==================================================
# 1. FASTAPI APPLICATION
# ==================================================

app = FastAPI(
    title="GlassBox ML Engine API",
    description="A neural network API for training and visualizing decision boundaries.",
    version="2.0.0",
)


# ==================================================
# 2. CORS CONFIGURATION
# ==================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==================================================
# 3. REQUEST SCHEMAS
# ==================================================

class Point(BaseModel):
    x1: float = Field(ge=0.0, le=1.0)
    x2: float = Field(ge=0.0, le=1.0)
    label: float = Field(ge=0.0, le=1.0)


class TrainRequest(BaseModel):
    points: List[Point] = Field(min_length=1)
    epochs: int = Field(default=200, ge=1, le=10000)
    learning_rate: float = Field(default=0.5, gt=0.0, le=10.0)


class ResetRequest(BaseModel):
    hidden_dim: int = Field(default=8, ge=1, le=256)
    learning_rate: float = Field(default=0.5, gt=0.0, le=10.0)


# ==================================================
# 4. MODEL INITIALIZATION
# ==================================================

DEFAULT_HIDDEN_DIM = 8
DEFAULT_LEARNING_RATE = 0.5


def create_model(
    hidden_dim: int = DEFAULT_HIDDEN_DIM,
    learning_rate: float = DEFAULT_LEARNING_RATE,
) -> GlassBoxMLP:
    return GlassBoxMLP(
        input_dim=2,
        hidden_dim=hidden_dim,
        output_dim=1,
        learning_rate=learning_rate,
    )


model = create_model()


# ==================================================
# 5. HOME / STATUS ENDPOINT
# ==================================================

@app.get("/")
def root():
    return {
        "status": "online",
        "message": "GlassBox ML Engine API is active!",
        "version": "2.0.0",
    }


@app.get("/health")
def health_check():
    return {"status": "healthy"}


# ==================================================
# 6. RESET MODEL
# ==================================================

@app.post("/reset")
def reset_model(req: ResetRequest):
    global model

    model = create_model(
        hidden_dim=req.hidden_dim,
        learning_rate=req.learning_rate,
    )

    return {
        "status": "success",
        "message": "Model re-initialized successfully",
        "hidden_dim": req.hidden_dim,
        "learning_rate": req.learning_rate,
    }


# ==================================================
# 7. TRAIN MODEL
# ==================================================

@app.post("/train")
def train_model(req: TrainRequest):
    global model

    # Validate binary classification labels
    if any(point.label not in (0.0, 1.0) for point in req.points):
        raise HTTPException(
            status_code=422,
            detail="Labels must be either 0 or 1.",
        )

    X = np.array(
        [[point.x1, point.x2] for point in req.points],
        dtype=float,
    )

    y = np.array(
        [[point.label] for point in req.points],
        dtype=float,
    )

    model.lr = req.learning_rate

    history = []

    for epoch in range(1, req.epochs + 1):
        loss, accuracy = model.train_epoch(X, y)

        history.append(
            {
                "epoch": epoch,
                "loss": float(loss),
                "accuracy": float(accuracy),
            }
        )

    return {
        "status": "success",
        "epochs_run": req.epochs,
        "final_loss": history[-1]["loss"],
        "final_accuracy": history[-1]["accuracy"],
        "history": history,
    }


# ==================================================
# 8. PREDICT
# ==================================================

@app.post("/predict")
def predict(points: List[Point]):
    if not points:
        raise HTTPException(
            status_code=400,
            detail="At least one point is required.",
        )

    X = np.array(
        [[point.x1, point.x2] for point in points],
        dtype=float,
    )

    probabilities = model.forward(X)

    predictions = []

    for point, probability in zip(points, probabilities):
        prob = float(probability[0])

        predictions.append(
            {
                "x1": point.x1,
                "x2": point.x2,
                "probability": prob,
                "predicted_class": int(prob >= 0.5),
            }
        )

    return {
        "status": "success",
        "predictions": predictions,
    }


# ==================================================
# 9. DECISION BOUNDARY
# ==================================================

@app.get("/decision-boundary")
def get_decision_boundary(
    resolution: int = Query(default=40, ge=2, le=200),
):
    x_range = np.linspace(0.0, 1.0, resolution)
    y_range = np.linspace(0.0, 1.0, resolution)

    # Generate all grid points at once
    xx, yy = np.meshgrid(x_range, y_range)

    grid_points = np.column_stack(
        (xx.ravel(), yy.ravel())
    )

    probabilities = model.forward(grid_points)

    grid = probabilities.reshape(
        resolution,
        resolution,
    ).tolist()

    return {
        "resolution": resolution,
        "x_range": x_range.tolist(),
        "y_range": y_range.tolist(),
        "grid": grid,
        "threshold": 0.5,
    }


# ==================================================
# 10. HIDDEN-LAYER HYPERPLANES
# ==================================================

@app.get("/hyperplanes")
def get_hyperplanes():
    W1 = model.W1
    b1 = model.b1.flatten()

    lines = []

    hidden_dim = W1.shape[1]

    for i in range(hidden_dim):
        w1 = float(W1[0, i])
        w2 = float(W1[1, i])
        bias = float(b1[i])

        # Hidden neuron boundary:
        # w1*x1 + w2*x2 + bias = 0
        lines.append(
            {
                "neuron": i,
                "w1": w1,
                "w2": w2,
                "b": bias,
            }
        )

    return {
        "status": "success",
        "count": len(lines),
        "lines": lines,
    }

