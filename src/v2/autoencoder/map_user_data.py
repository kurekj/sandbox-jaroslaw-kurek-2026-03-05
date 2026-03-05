# mypy: ignore-errors
from typing import Any, Optional, TypeVar, Union

import dash
import dash_bootstrap_components as dbc
import numpy as np
import numpy.typing as npt
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, State, callback, dcc, html
from dash.exceptions import PreventUpdate

# Type hints
T = TypeVar("T")
PropertyDF = pd.DataFrame
PropertyId = Union[int, str]


class UserPropertyMap:
    def __init__(
        self,
        property_df: PropertyDF,
        user_property_scores: npt.NDArray[np.float64],
        viewed_properties: PropertyDF,
        data_columns: list[str] = [
            "offer_id",
            "city_name",
            "area",
            "rooms",
            "floor",
            "bathrooms",
            "normalize_price_m2",
            "normalize_price",
            "facilities",
            "natural_sites",
            "additional_area_type",
            "additional_area_area",
        ],
    ) -> None:
        """
        Initialize the UserPropertyMap with property data, user-property scores, and viewed properties.

        Args:
            property_df: DataFrame with property data (must contain property_id, lat, lon columns)
            user_property_scores: Numpy array with scores for each property (same length as property_df)
            viewed_properties: DataFrame with properties that the user has viewed (same columns as property_df)
            data_columns: list of property columns to display in hover information
        """
        self.data_columns = data_columns

        # Validate inputs
        required_columns = ["property_id", "lat", "lon"]
        if not all(col in property_df.columns for col in required_columns):
            raise ValueError(f"property_df must contain these columns: {required_columns}")

        if len(property_df) != len(user_property_scores):
            raise ValueError("user_property_scores must have same length as property_df")

        if not all(col in viewed_properties.columns for col in required_columns):
            raise ValueError(f"viewed_properties must contain these columns: {required_columns}")

        self.property_df = property_df
        self.user_property_scores = user_property_scores
        self.viewed_properties = viewed_properties

        # Create property ID to index mapping for fast lookups
        self.id_to_idx = {id_val: idx for idx, id_val in enumerate(property_df["property_id"])}
        self.idx_to_id = {idx: id_val for idx, id_val in enumerate(property_df["property_id"])}

        # Get the range of scores
        self.min_score = np.min(user_property_scores)
        self.max_score = np.max(user_property_scores)

    def get_properties_with_scores(self, score_threshold: float) -> PropertyDF:
        """
        Get properties with scores based on the threshold.

        Args:
            score_threshold: Score threshold - properties with score >= threshold are considered relevant

        Returns:
            DataFrame with properties and their scores
        """
        # Create result DataFrame
        result = self.property_df.copy()

        # Add score column
        result["score"] = self.user_property_scores

        # Add flag for properties meeting the threshold
        result["above_threshold"] = result["score"] >= score_threshold

        # # Get viewed property IDs
        # viewed_ids = self.viewed_properties["property_id"].unique().tolist()
        # print(f"Viewed IDs: {viewed_ids}")

        # add viewed properties to the result DataFrame with score 1 if they are not present in the result
        # DataFrame
        viewed_ids = self.viewed_properties["property_id"].unique().tolist()
        for id_val in viewed_ids:
            if id_val not in self.id_to_idx:
                # Create a new row for the viewed property
                row = self.viewed_properties[self.viewed_properties["property_id"] == id_val].copy()

                result = pd.concat([result, row], ignore_index=True)
                self.id_to_idx[id_val] = len(result) - 1
                self.idx_to_id[len(result) - 1] = id_val
                # Add flags for viewed properties

        # fill nan score with 1
        result["score"] = result["score"].fillna(1)
        result["above_threshold"] = result["score"] >= score_threshold

        # Add flag for viewed properties
        result["viewed"] = result["property_id"].isin(viewed_ids)

        return result


def create_user_dashboard(
    property_df: PropertyDF,
    user_property_scores: npt.NDArray[np.float64],
    viewed_properties: PropertyDF,
) -> dash.Dash:
    """
    Create and return a Dash app with the user property recommendation map.

    Args:
        property_df: DataFrame with property data
        user_property_scores: Numpy array with scores for each property (same length as property_df)
        viewed_properties: DataFrame with properties that the user has viewed (same columns as property_df)

    Returns:
        Configured Dash application
    """
    map_handler = UserPropertyMap(property_df, user_property_scores, viewed_properties)

    # Initialize the Dash app
    app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

    # Layout components
    controls = dbc.Card(
        [
            dbc.CardHeader("Controls"),
            dbc.CardBody(
                [
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Label("Score Threshold: "),
                                    html.Span(id="threshold-display", style={"marginLeft": "10px"}),
                                ],
                                style={"display": "flex", "alignItems": "center"},
                            ),
                            dcc.Slider(
                                id="score-threshold",
                                min=map_handler.min_score,
                                max=map_handler.max_score,
                                value=(map_handler.min_score + map_handler.max_score) / 2,  # Default to middle value
                                marks={
                                    map_handler.min_score: f"{map_handler.min_score:.2f}",
                                    map_handler.max_score: f"{map_handler.max_score:.2f}",
                                },
                            ),
                        ],
                        className="mb-4",
                    ),
                    html.Div(
                        [
                            dbc.Checkbox(
                                id="hide-below-threshold",
                                label="Hide properties below threshold",
                                value=False,  # Initially show all points
                                className="mr-2",
                            ),
                        ],
                        className="mb-4",
                    ),
                    html.Hr(),
                    html.Div(id="user-property-info", className="mt-4"),
                ]
            ),
        ],
        className="mb-4",
    )

    # App layout
    app.layout = dbc.Container(
        [
            html.H1("User Property Recommendation Map", className="my-4"),
            dbc.Row(
                [
                    dbc.Col(controls, md=3),
                    dbc.Col(
                        [
                            dcc.Graph(
                                id="map-graph",
                                style={"height": "80vh"},
                                config={
                                    "scrollZoom": True,
                                    "displayModeBar": True,
                                    "modeBarButtonsToRemove": ["lasso2d", "select2d"],
                                },
                            ),
                            dcc.Store(id="clicked-property"),
                            dcc.Store(id="map-view-state", data={"is_initial_view": True}),
                        ],
                        md=9,
                    ),
                ]
            ),
        ],
        fluid=True,
    )

    @callback(Output("threshold-display", "children"), Input("score-threshold", "value"))
    def update_threshold_display(threshold):
        return f"{threshold:.4f}"

    # Callback to update map based on threshold
    @callback(
        Output("map-graph", "figure"),
        Output("user-property-info", "children"),
        Output("map-view-state", "data"),
        Input("score-threshold", "value"),
        Input("hide-below-threshold", "value"),
        Input("clicked-property", "data"),
        State("map-view-state", "data"),
    )
    def update_map(
        threshold: float,
        hide_below_threshold: bool,
        clicked_data: Optional[dict[str, Any]],
        view_state: dict[str, Any],
    ) -> tuple:
        # Get properties with scores
        properties = map_handler.get_properties_with_scores(threshold)

        # Create map figure
        fig = go.Figure()

        # Add scatter map for properties below threshold
        below_threshold = properties[~properties["above_threshold"]]
        if not below_threshold.empty and not hide_below_threshold:
            # Convert customdata to a list of lists
            below_customdata = [
                [float(row["score"])] + [row[col] for col in map_handler.data_columns]
                for _, row in below_threshold.iterrows()
            ]

            fig.add_trace(
                go.Scattermapbox(
                    lat=below_threshold["lat"].tolist(),
                    lon=below_threshold["lon"].tolist(),
                    mode="markers",
                    marker=dict(
                        size=10,
                        color="gray",
                        opacity=0.2,
                    ),
                    text=below_threshold["property_id"].tolist(),
                    hovertemplate=(
                        "<b>Property %{text}</b><br>"
                        + "Score: %{customdata[0]:.2f}<br>"
                        + "<br>".join(
                            [f"{col}: %{{customdata[{i + 1}]}}" for i, col in enumerate(map_handler.data_columns)]
                        )
                        + "<extra></extra>"
                    ),
                    customdata=below_customdata,
                    name="Below Threshold",
                )
            )

        # Add scatter map for properties above threshold (colored by score)
        above_threshold = properties[properties["above_threshold"] & ~properties["viewed"]]
        if not above_threshold.empty:
            # Convert customdata to a list of lists
            above_customdata = [
                [float(row["score"])] + [row[col] for col in map_handler.data_columns]
                for _, row in above_threshold.iterrows()
            ]

            fig.add_trace(
                go.Scattermapbox(
                    lat=above_threshold["lat"].tolist(),
                    lon=above_threshold["lon"].tolist(),
                    mode="markers",
                    marker=dict(
                        size=14,
                        color=above_threshold["score"].tolist(),
                        colorscale="Viridis",
                        colorbar=dict(
                            title="Score",
                            x=0.9,
                        ),
                        cmin=map_handler.min_score,
                        cmax=map_handler.max_score,
                    ),
                    text=above_threshold["property_id"].tolist(),
                    hovertemplate=(
                        "<b>Property %{text}</b><br>"
                        + "Score: %{customdata[0]:.2f}<br>"
                        + "<br>".join(
                            [f"{col}: %{{customdata[{i + 1}]}}" for i, col in enumerate(map_handler.data_columns)]
                        )
                        + "<extra></extra>"
                    ),
                    customdata=above_customdata,
                    name="Recommended Properties",
                )
            )

        # Add viewed properties (highlighted)
        viewed = properties[properties["viewed"]]
        if not viewed.empty:
            # Convert customdata to a list of lists
            viewed_customdata = [
                [float(row["score"])] + [row[col] for col in map_handler.data_columns] for _, row in viewed.iterrows()
            ]

            fig.add_trace(
                go.Scattermapbox(
                    lat=viewed["lat"].tolist(),
                    lon=viewed["lon"].tolist(),
                    mode="markers",
                    marker=dict(
                        size=18,
                        color="red",
                        # symbol="star",
                    ),
                    text=viewed["property_id"].tolist(),
                    hovertemplate=(
                        "<b>Property %{text} (Viewed)</b><br>"
                        + "Score: %{customdata[0]:.2f}<br>"
                        + "<br>".join(
                            [f"{col}: %{{customdata[{i + 1}]}}" for i, col in enumerate(map_handler.data_columns)]
                        )
                        + "<extra></extra>"
                    ),
                    customdata=viewed_customdata,
                    name="Viewed Properties",
                )
            )

        # Set map layout
        layout_args = {
            "style": "carto-positron",
        }

        # Only set center and zoom on initial view
        if view_state.get("is_initial_view", True):
            # Calculate the center of all property points
            all_lats = property_df["lat"].values
            all_lons = property_df["lon"].values

            # Set center to average of all points
            center_lat = np.mean(all_lats)
            center_lon = np.mean(all_lons)

            layout_args.update(
                {
                    "center": {"lat": center_lat, "lon": center_lon},
                    "zoom": 6,
                }
            )
        else:
            # Get the current view state
            center = view_state.get("center", {"lat": 52.23, "lon": 21.01})
            zoom = view_state.get("zoom", 6)

            layout_args.update(
                {
                    "center": center,
                    "zoom": zoom,
                }
            )

        # Update map layout
        fig.update_layout(
            mapbox=layout_args,
            margin=dict(l=0, r=0, t=0, b=0),
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                bgcolor="rgba(255, 255, 255, 0.8)",
            ),
            height=1000,
        )

        # Create info panel for properties
        # Count of properties above threshold
        above_count = sum(properties["above_threshold"])

        # Count of viewed properties
        viewed_count = sum(properties["viewed"])

        # Create info panel
        info_panel = [
            html.H5("Property Recommendations"),
            html.Hr(),
            html.Div(
                [
                    html.Strong("Properties above threshold: "),
                    html.Span(f"{above_count}"),
                ]
            ),
            html.Div(
                [
                    html.Strong("Viewed properties: "),
                    html.Span(f"{viewed_count}"),
                ]
            ),
            html.Hr(),
            html.H5("Top Recommended Properties:"),
            html.Div(
                [
                    html.Ul(
                        [
                            html.Li(
                                [
                                    html.Strong(f"Property {row['property_id']} - Score: {row['score']:.2f}"),
                                    html.Ul([html.Li(f"{col}: {row[col]}") for col in map_handler.data_columns]),
                                ]
                            )
                            for _, row in properties[~properties["viewed"]]
                            .sort_values("score", ascending=False)
                            .head(10)
                            .iterrows()
                        ],
                        style={"overflowY": "scroll", "height": "300px"},
                    ),
                ]
            ),
            html.Hr(),
            html.H5("Viewed Properties:"),
            html.Div(
                [
                    html.Ul(
                        [
                            html.Li(
                                [
                                    html.Strong(f"Property {row['property_id']} - Score: {row['score']:.2f}"),
                                    html.Ul([html.Li(f"{col}: {row[col]}") for col in map_handler.data_columns]),
                                ]
                            )
                            for _, row in properties[properties["viewed"]]
                            .sort_values("score", ascending=False)
                            .iterrows()
                        ],
                        style={"overflowY": "scroll", "height": "300px"},
                    ),
                ]
            ),
        ]

        # Set view state to no longer be initial view
        new_view_state = {"is_initial_view": False}

        # Return the figure, info panel, and view state
        return fig, info_panel, new_view_state

    # Callback to handle map clicks
    @callback(
        Output("clicked-property", "data"),
        Input("map-graph", "clickData"),
    )
    def handle_map_click(click_data: Optional[dict[str, Any]]) -> Optional[dict[str, Any]]:
        if click_data is None:
            raise PreventUpdate

        # Extract clicked coordinates
        point = click_data["points"][0]

        # Handle case where clicked point is not a property
        if "text" not in point:
            raise PreventUpdate

        # Get the property ID from the clicked point
        property_id = point["text"]

        # Return clicked property data
        return {"property_id": property_id}

    return app
