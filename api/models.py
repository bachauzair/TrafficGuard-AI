from pydantic import BaseModel, Field

class SeverityRequest(BaseModel):
    Temperature_F: float = Field(default=60.0, alias="Temperature(F)")
    Visibility_mi: float = Field(default=10.0, alias="Visibility(mi)")
    Wind_Speed_mph: float = Field(default=10.0, alias="Wind_Speed(mph)")
    Humidity_pct: float = Field(default=50.0, alias="Humidity(%)")
    Hour: int = 12
    DayOfWeek: int = 1
    Month: int = 6
    Weather_Category: str = "Clear"
    Season: int = 1
    Is_Weekend: int = 0
    Junction: int = 0
    Traffic_Signal: int = 0
    Crossing: int = 0
    Station: int = 0
    Sunrise_Sunset: str = "Day"

class SeverityResponse(BaseModel):
    predicted_severity: int
    severity_label: str
    confidence: float
    top_factors: list
