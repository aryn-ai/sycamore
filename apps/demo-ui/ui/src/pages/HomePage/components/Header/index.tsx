import {
  Burger,
  Container,
  Group,
  Header,
  Image,
  MediaQuery,
  Text,
  useMantineTheme,
} from "@mantine/core";
import { Dispatch, SetStateAction } from "react";
import { Settings } from "../../../../Types";
import { useMediaQuery } from "@mantine/hooks";

export const HeaderComponent = ({
  navBarOpened,
  setNavBarOpened,
  settings,
}: {
  navBarOpened: boolean;
  setNavBarOpened: Dispatch<SetStateAction<boolean>>;
  settings: Settings;
}) => {
  const theme = useMantineTheme();
  const mobileScreen = useMediaQuery(`(max-width: ${theme.breakpoints.sm})`);

  return (
    <Header
      height={80}
      sx={(theme) => ({
        display: "flex",
        alignItems: "center",
        paddingLeft: theme.spacing.md,
        paddingRight: theme.spacing.md,
      })}
    >
      <Group w="100%">
        <MediaQuery largerThan="sm" styles={{ display: "none" }}>
          <Burger
            opened={navBarOpened}
            onClick={() => setNavBarOpened((o) => !o)}
            size="sm"
            maw="0.5rem"
            mr="xl"
          />
        </MediaQuery>
        <Image
          width={mobileScreen ? "18em" : "24em"}
          src="./SycamoreDemoQueryUI_Logo.png"
        />
        {!mobileScreen && (
          <Container pos="absolute" right="0rem">
            <Text fz="xs" c="dimmed">
              index: {settings.openSearchIndex}
            </Text>
            <Text fz="xs" c="dimmed">
              llm model: {settings.modelName}
            </Text>
            <Text fz="xs" c="dimmed">
              llm model id: {settings.modelId}
            </Text>
          </Container>
        )}
      </Group>
    </Header>
  );
};
