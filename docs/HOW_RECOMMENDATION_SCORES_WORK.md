# How Property Recommendation Scores Work

## Overview

This document explains how our property recommendation system calculates personalized scores for users when they view properties. The system uses artificial intelligence to understand what types of properties a user likes and recommends similar ones.

## The Big Picture

Imagine you're looking at a property listing, and the system needs to decide how much you might like it. To do this, it looks at two things:

1. **Properties you've applied for or interacted with before** (your application/lead history)
2. **The property you're currently looking at**

The system then calculates a "similarity score" that tells us how well the current property matches your preferences based on your past application behavior.

**Note:** The system specifically uses your application history (when you've submitted leads or applications for properties) rather than just viewing history, as this represents stronger interest signals. Yet, it's a subject to change as we explore different approaches to recommendation scoring like including views, search and inquiry data.

## How It Works Step by Step

### Step 1: Converting Properties into "Digital Fingerprints" (Embeddings)

Every property in our database gets converted into what we call an "embedding" - think of it as a unique digital fingerprint that captures all the important characteristics of a property.

#### What Information Goes Into These Fingerprints?

The system looks at many features of each property:

**Location Features:**

- Geographic coordinates (longitude and latitude)
- Neighborhood characteristics
- Nearby points of interest (shops, schools, hospitals, transport, etc.)

**Property Characteristics:**

- Size (area in square meters)
- Number of rooms and bathrooms
- Floor number
- Kitchen type
- Additional areas (balcony, garden, etc.)
- Property type (apartment, house, etc.)

**Pricing Information:**

- Normalized price per square meter
- Total normalized price

**Surroundings:**

- Count and distance to various amenities:
  - Shopping centers
  - Restaurants and food places
  - Schools and universities
  - Healthcare facilities
  - Entertainment venues
  - Sports facilities
  - Public transportation

#### How These Fingerprints Are Created

We use an AI model called an **autoencoder** to create these digital fingerprints. Here's how it works:

1. **Training Data Collection**: The system analyzes thousands of existing properties with all their features
2. **Learning Patterns**: The AI learns which features tend to go together (for example, properties in city centers might have smaller areas but better transport access)
3. **Compression**: The AI learns to compress all this information into a much smaller "fingerprint" (32 numbers) that still captures the essence of what makes each property unique
4. **Quality Check**: The AI is trained to reconstruct the original property features from just the fingerprint to ensure no important information is lost

Think of it like teaching a computer to recognize the "DNA" of properties - it learns what makes properties similar or different from each other.

### Step 2: Understanding Your Preferences

When you submit applications for properties or interact with them through our platform, the system keeps track of:

- Which properties you've applied for or shown strong interest in
- When you applied for them (more recent applications are considered more important)
- The full details and features of those properties

Each property in your application history gets its own digital fingerprint, just like the new property you're considering.

### Step 3: The Smart Matching Process (Attention Mechanism)

This is where the magic happens. The system doesn't just average your past preferences - it's much smarter than that.

For each new property you're considering, the system:

1. **Compares the new property to each property in your application history** to see how similar they are
2. **Gives more weight to recent applications** (properties you applied for yesterday matter more than those from last month)
3. **Creates a personalized preference profile** specifically for this new property

#### The Attention Mechanism Explained

Here's the clever part: the system creates a different "version" of your preferences for each new property you look at. It does this by:

1. **Looking at similarities**: For each property in your application history, it calculates how similar it is to the new property using dot product similarity
2. **Applying time weighting**: Recent applications in your history get more importance through a configurable time decay factor
3. **Creating attention weights**: The system combines similarity scores with time weights, then applies a softmax function with temperature scaling to create attention weights. Properties in your history that are more similar to the current property get higher "attention" - meaning they influence the final score more
4. **Building a custom profile**: Instead of using a generic profile of your preferences, it creates a specific preference profile tailored to the type of property you're currently viewing by computing a weighted average of your historical property embeddings

For example:

- If you're looking at a city center apartment, the system will pay more attention to other city center apartments in your application history
- If you're looking at a suburban house, it will focus more on the suburban properties you've applied for before
- This makes the recommendations much more contextual and relevant

### Step 4: Calculating the Final Score

The final step involves:

1. **Creating your personalized preference profile** for this specific property type using the attention-weighted combination of your historical preferences
2. **Comparing the new property** to this tailored preference profile using dot product similarity
3. **Generating a similarity score** between -1 and 1, where:
   - **1** means the property is extremely similar to your preferences
   - **0** means the property is neutral (neither similar nor dissimilar)
   - **-1** means the property is very different from what you typically apply for

**Technical note:** The final score is computed as the dot product between the property's embedding and your personalized weighted preference embedding, which ensures that if both embeddings are normalized, the score will be bounded within [-1, 1].

## Why This Approach Works Well

### Personalization

- Each user gets recommendations based on their unique application history
- The system learns your preferences automatically from your actual application behavior

### Context-Aware

- The attention mechanism ensures that recommendations are relevant to the type of property you're currently viewing
- You get different recommendations when looking at apartments vs. houses vs. commercial properties

### Time-Sensitive

- Recent applications matter more than old ones, so your evolving tastes are captured
- If you start applying for different types of properties, the system adapts quickly

### Handles Complexity

- The AI can understand complex relationships between property features
- It can recognize that "good location" might mean different things for different property types

## Technical Benefits

### Scalability

- Once a property's fingerprint is calculated, it can be reused for all users
- New properties can be immediately recommended to all users without retraining the entire system

### Caching

- Scores are cached to avoid recalculating the same user-property pairs repeatedly
- This makes the system fast and responsive

### Continuous Learning

- The embedding model can be retrained periodically with new data to improve over time
- New property features can be easily added to the system

## Summary

In simple terms, our recommendation system works like a very sophisticated matchmaker:

1. It creates a unique "personality profile" for every property using AI
2. It remembers what types of properties you've actually applied for
3. When you look at a new property, it creates a personalized version of your preferences specifically for that type of property using an attention mechanism
4. It then calculates how well the new property matches your personalized preferences using mathematical similarity measures
5. The result is a score that tells you how likely you are to be interested in this property based on your past application behavior

This approach ensures that you see the most relevant properties while browsing, making your property search experience more efficient and personalized.
