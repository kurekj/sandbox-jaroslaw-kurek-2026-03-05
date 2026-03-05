# mypy: ignore-errors
from typing import TypeVar, cast

import dash
import dash_bootstrap_components as dbc  # type: ignore
import numpy as np
import numpy.typing as npt
import pandas as pd
import plotly.graph_objects as go  # type: ignore
from dash import Input, Output, State, callback, dcc, html
from dash.exceptions import PreventUpdate

# Type hints
T = TypeVar("T")
PropertyDF = pd.DataFrame
SimilarityMatrix = npt.NDArray[np.float64]
PropertyId = int | str


class RealEstateMap:
    def __init__(
        self,
        property_df: PropertyDF,
        similarity_matrix: SimilarityMatrix,
        color_clusters: bool = False,
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
            "cluster",
            "recommendation",
        ],
    ) -> None:
        """
        Initialize the RealEstateMap with property data and similarity matrix.

        Args:
            property_df: DataFrame with property data (must contain property_id, lat, lon columns)
            similarity_matrix: Square numpy array with similarity distances between properties
        """
        self.data_columns = data_columns

        # Validate inputs
        required_columns = ["property_id", "lat", "lon"]
        if not all(col in property_df.columns for col in required_columns):
            raise ValueError(f"property_df must contain these columns: {required_columns}")

        if len(property_df) != similarity_matrix.shape[0] or similarity_matrix.shape[0] != similarity_matrix.shape[1]:
            raise ValueError("Similarity matrix dimensions must match number of properties")

        self.property_df = property_df
        self.similarity_matrix = similarity_matrix

        # Get the maximum similarity value for slider range
        self.max_similarity = np.max(similarity_matrix)

        # Create property ID to index mapping for fast lookups
        self.id_to_idx = {id_val: idx for idx, id_val in enumerate(property_df["property_id"])}
        self.idx_to_id = {idx: id_val for idx, id_val in enumerate(property_df["property_id"])}

        self.color_clusters = color_clusters

    def get_similar_properties(self, reference_id: PropertyId, threshold: float) -> pd.DataFrame:
        """
        Get properties similar to the reference property within the threshold.

        Args:
            reference_id: ID of the reference property
            threshold: Similarity threshold - properties with similarity <= threshold are considered similar

        Returns:
            DataFrame with properties and similarity scores
        """
        try:
            ref_idx = self.id_to_idx[reference_id]
        except KeyError:
            raise ValueError(f"Property ID {reference_id} not found")

        # Get similarity scores for all properties relative to reference
        similarities = self.similarity_matrix[ref_idx]

        # Create result DataFrame with similarity score
        result = self.property_df.copy()
        result["similarity"] = similarities

        # Add flag for properties meeting threshold (lower value = more similar)
        if self.color_clusters:
            result["below_threshold"] = True
        else:
            result["below_threshold"] = result["similarity"] <= threshold

        return result

    def get_reference_property(self, reference_id: PropertyId) -> pd.Series:  # type: ignore
        """Get the reference property data by ID"""
        try:
            return self.property_df.loc[self.property_df["property_id"] == reference_id].iloc[0]
        except (KeyError, IndexError):
            raise ValueError(f"Property ID {reference_id} not found")

    def get_property_id_from_coords(self, lat: float, lon: float) -> list[PropertyId]:
        """Get property IDs at the given coordinates"""
        mask = (self.property_df["lat"] == lat) & (self.property_df["lon"] == lon)
        return self.property_df.loc[mask, "property_id"].tolist()


# Dashboard Implementation
def create_dashboard(
    property_df: PropertyDF,
    similarity_matrix: SimilarityMatrix,
    color_clusters: bool = False,
) -> dash.Dash:
    """
    Create and return a Dash app with the real estate similarity map.

    Args:
        property_df: DataFrame with property data
        similarity_matrix: Similarity matrix (distances between properties)
        color_clusters: Flag to enable color clustering on the map

    Returns:
        Configured Dash application
    """
    map_handler = RealEstateMap(property_df, similarity_matrix, color_clusters)

    # Get the maximum similarity value for the slider
    max_similarity = map_handler.max_similarity

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
                            html.Label("Select Reference Property:"),
                            dcc.Dropdown(
                                id="property-dropdown",
                                options=[
                                    {"label": f"Property {p_id}", "value": p_id} for p_id in property_df["property_id"]
                                ],
                                value=property_df["property_id"].iloc[0],
                                clearable=False,
                            ),
                        ],
                        className="mb-4",
                    ),
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Label("Similarity Threshold: "),
                                    html.Span(id="threshold-display", style={"marginLeft": "10px"}),
                                ],
                                style={"display": "flex", "alignItems": "center"},
                            ),
                            dcc.Slider(
                                id="similarity-threshold",
                                min=0,
                                max=round(max_similarity, 2),
                                # step=round(max_similarity / 100, 2),  # 100 steps
                                value=round(max_similarity / 10, 2),  # Default to middle value
                            ),
                        ],
                        className="mb-4",
                    ),
                    html.Div(
                        [
                            dbc.Checkbox(
                                id="hide-above-threshold",
                                label="Hide points above threshold",
                                value=False,  # Initially show all points
                                className="mr-2",
                            ),
                        ],
                        className="mb-4",
                    ),
                    html.Div(
                        [
                            dbc.Checkbox(
                                id="hide-duplicate-offers",
                                label="Hide properties with same offer_id as reference",
                                value=False,  # Initially show all points
                                className="mr-2",
                            ),
                        ],
                        className="mb-4",
                    ),
                    html.Hr(),
                    html.Div(id="selected-property-info", className="mt-4"),
                ]
            ),
        ],
        className="mb-4",
    )

    # App layout
    app.layout = dbc.Container(
        [
            html.H1("Real Estate Similarity Map", className="my-4"),
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

    @callback(Output("threshold-display", "children"), Input("similarity-threshold", "value"))
    def update_threshold_display(threshold):
        return f"{threshold:.4f}"

    # Callback to update map based on reference property and threshold
    @callback(
        Output("map-graph", "figure"),
        Output("selected-property-info", "children"),
        Output("map-view-state", "data"),
        Input("property-dropdown", "value"),
        Input("similarity-threshold", "value"),
        Input("hide-above-threshold", "value"),
        Input("hide-duplicate-offers", "value"),
        Input("clicked-property", "data"),
        State("map-view-state", "data"),
    )
    def update_map(
        dropdown_property_id: PropertyId,
        threshold: float,
        hide_above_threshold: bool,
        hide_duplicate_offers: bool,
        clicked_data: dict | None,
        view_state: dict,
    ) -> tuple[dict, list]:
        # Determine which input triggered the callback
        ctx = dash.callback_context
        trigger_id = ctx.triggered[0]["prop_id"].split(".")[0] if ctx.triggered else ""

        # If map was clicked, update reference property
        if trigger_id == "clicked-property" and clicked_data is not None:
            property_id = clicked_data["property_id"]
            # Only update if property exists
            if property_id in map_handler.id_to_idx:
                property_id = cast(PropertyId, property_id)  # Explicit cast for type checking
            else:
                # Fall back to dropdown value
                property_id = dropdown_property_id
        else:
            property_id = dropdown_property_id

        if isinstance(property_id, str):
            property_id = int(property_id)

        print(f"Property ID: {property_id}")

        # Get similar properties based on threshold
        similar_props = map_handler.get_similar_properties(property_id, threshold)

        # Get reference property
        ref_prop = map_handler.get_reference_property(property_id)

        # Filter out properties with the same offer_id as reference property if enabled
        if hide_duplicate_offers:
            # Get the offer_id of the reference property
            ref_offer_id = ref_prop["offer_id"]

            # Keep the reference property and properties with different offer_ids
            ref_mask = similar_props["property_id"] == property_id
            same_offer_mask = (similar_props["offer_id"] == ref_offer_id) & (~ref_mask)

            # Filter out properties with the same offer_id as reference (except the reference itself)
            similar_props = similar_props[~same_offer_mask | ref_mask]

        # Determine if this is the initial view
        is_initial_view = view_state.get("is_initial_view", True)

        # Create map figure
        fig = go.Figure()

        # Add scatter map for properties below threshold (gray)
        above_threshold = similar_props[~similar_props["below_threshold"]]
        if not above_threshold.empty and not hide_above_threshold:
            # Convert customdata to a simple list of lists to ensure serialization
            above_customdata = [
                [float(row["similarity"])] + [row[col] for col in map_handler.data_columns]
                for _, row in above_threshold.iterrows()
            ]

            fig.add_trace(
                go.Scattermap(  # Use Scattermap instead of deprecated Scattermapbox
                    lat=above_threshold["lat"].tolist(),  # Convert to Python list
                    lon=above_threshold["lon"].tolist(),  # Convert to Python list
                    mode="markers",
                    marker=dict(
                        size=10,
                        color="gray",
                        opacity=0.2,
                    ),
                    text=above_threshold["property_id"].tolist(),  # Convert to Python list
                    hovertemplate=(
                        "<b>Property %{text}</b><br>"
                        + "Similarity: %{customdata[0]:.2f}<br>"
                        + "<br>".join(
                            [f"{col}: %{{customdata[{i + 1}]}}" for i, col in enumerate(map_handler.data_columns)]
                        )
                        + "<extra></extra>"
                    ),
                    customdata=above_customdata,
                    name="Below Threshold",
                )
            )

        # Add scatter map for properties above threshold (colored by similarity)
        below_threshold = similar_props[
            similar_props["below_threshold"] & (similar_props["property_id"] != property_id)
        ]
        if not below_threshold.empty:
            # Convert customdata to a simple list of lists to ensure serialization
            below_customdata = [
                [float(row["similarity"])] + [row[col] for col in map_handler.data_columns]
                for _, row in below_threshold.iterrows()
            ]
            if not color_clusters:
                marker_config = dict(
                    size=14,
                    color=below_threshold["similarity"].tolist(),  # Convert to Python list
                    # color_continuous_scale="Virdis",  # Updated color scale,
                    colorscale="Viridis",
                    colorbar=dict(
                        title="Similarity",
                        x=0.9,
                    ),
                )
            else:
                # color markers by cluster id where -1 are outliers, and >=0 are clusters
                marker_config = dict(
                    size=14,
                    # color=below_threshold["cluster"].tolist(),  # Convert to Python list
                    color=["red" if row.recommendation else row.cluster for row in below_threshold.itertuples()],
                    colorscale="Viridis",
                    colorbar=dict(
                        title="Cluster ID",
                        x=0.9,
                    ),
                    # for recommendation True show points as cross, but only for them rest is circle
                    # symbol="star",
                    # symbol=["cross" if rec else "circle" for rec in below_threshold["recommendation"]],
                    showscale=True,
                    cmin=-1,
                    cmax=below_threshold["cluster"].max(),
                )

            fig.add_trace(
                go.Scattermap(  # Use Scattermap instead of deprecated Scattermapbox
                    lat=below_threshold["lat"].tolist(),  # Convert to Python list
                    lon=below_threshold["lon"].tolist(),  # Convert to Python list
                    mode="markers",
                    marker=marker_config,
                    text=below_threshold["property_id"].tolist(),  # Convert to Python list
                    hovertemplate=(
                        "<b>Property %{text}</b><br>"
                        + "Similarity: %{customdata[0]:.2f}<br>"
                        + "<br>".join(
                            [f"{col}: %{{customdata[{i + 1}]}}" for i, col in enumerate(map_handler.data_columns)]
                        )
                        + "<extra></extra>"
                    ),
                    customdata=below_customdata,
                    name="Similar Properties",
                )
            )

        # Add reference property (highlighted)
        ref_customdata = [[ref_prop[col] for col in map_handler.data_columns]]

        fig.add_trace(
            go.Scattermap(  # Use Scattermap instead of deprecated Scattermapbox
                lat=[float(ref_prop["lat"])],  # Ensure it's a float
                lon=[float(ref_prop["lon"])],  # Ensure it's a float
                mode="markers",
                marker=dict(
                    size=18,
                    color="red",
                    symbol="star",
                ),
                text=[ref_prop["property_id"]],
                hovertemplate=(
                    "<b>Property %{text} (Reference)</b><br>"
                    + "<br>".join([f"{col}: %{{customdata[{i}]}}" for i, col in enumerate(map_handler.data_columns)])
                    + "<extra></extra>"
                ),
                customdata=ref_customdata,
                name="Reference Property",
            )
        )

        layout_args = {
            "style": "carto-positron",
        }

        # Only set center and zoom on initial view
        if is_initial_view:
            # Calculate the center of all property points
            all_lats = property_df["lat"].values
            all_lons = property_df["lon"].values

            # Set center to average of all points
            center_lat = np.mean(all_lats)
            center_lon = np.mean(all_lons)

            # For Poland, set appropriate zoom
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

        # Set map layout
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

        # Create info panel for selected property
        # Create info panel for selected property
        property_info_elements = []
        for col in map_handler.data_columns:
            property_info_elements.append(html.Strong(f"{col}: "))
            property_info_elements.append(html.Span(f"{ref_prop[col]}"))
            property_info_elements.append(html.Br())

        info_panel = [
            html.H5(f"Reference Property: {property_id}"),
            html.Hr(),
            html.Div(property_info_elements),
            html.Hr(),
            html.Div(
                [
                    html.Strong("Similar Properties (count): "),
                    html.Span(f"{sum(similar_props['below_threshold']) - 1}"),  # -1 to exclude reference
                ]
            ),
            html.Hr(),
            html.H5("Similar Properties:"),
            html.Div(
                [
                    html.Ul(
                        [
                            html.Li(
                                [
                                    html.Strong(
                                        f"Property {row['property_id']} {row['offer_id']} - "
                                        + f"Similarity: {row['similarity']:.2f}"
                                    ),
                                    html.Ul([html.Li(f"{col}: {row[col]}") for col in map_handler.data_columns]),
                                ]
                            )
                            for _, row in similar_props.sort_values("similarity").head(25).iterrows()
                            if not hide_above_threshold or row["offer_id"] != ref_prop["offer_id"]
                        ],
                        style={"overflowY": "scroll", "height": "400px"},
                    )
                ]
            ),
        ]

        # Set view state to no longer be initial view
        new_view_state = {"is_initial_view": False}

        # Return a dictionary representation of the figure instead of the figure object
        return fig.to_dict(), info_panel, new_view_state

    # Callback to handle map clicks
    @callback(
        Output("clicked-property", "data"),
        Output("property-dropdown", "value"),
        Input("map-graph", "clickData"),
        State("property-dropdown", "value"),
    )
    def handle_map_click(click_data: dict | None, current_property_id: PropertyId) -> tuple[dict | None, PropertyId]:
        if click_data is None:
            raise PreventUpdate

        # Extract clicked coordinates
        point = click_data["points"][0]

        # Handle case where clicked point is not a property
        if "text" not in point:
            raise PreventUpdate

        # Get the property ID from the clicked point
        property_id = point["text"]

        # Update clicked property data store
        return {"property_id": property_id}, property_id

    return app
